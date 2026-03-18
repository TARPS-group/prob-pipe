"""
Discrete distributions backed by TFP.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .distribution import (
    TFPDistribution,
    Distribution,
    EmpiricalDistribution,
    Provenance,
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

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Bernoulli:
        if isinstance(other, Bernoulli):
            result = cls(
                probs=other.probs,
                logits=other.logits,
                name=name or other.name,
            )
            return result.with_source(
                Provenance("from_distribution", parents=(other,))
            )
        num_samples = kwargs.pop("num_samples", 1024)
        probs = other.mean()
        result = cls(probs=probs, name=name or other.name)
        return result.with_source(
            Provenance("from_distribution", parents=(other,))
        )


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

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Binomial:
        """Fit a Binomial from *other*.

        Keyword Args
        -------------
        total_count : int
            Number of trials. **Required** unless *other* is already a
            ``Binomial`` (in which case its ``total_count`` is reused).
        num_samples : int
            Samples drawn for moment-matching (default 1024).
        """
        if isinstance(other, Binomial):
            result = cls(
                total_count=other.total_count,
                probs=other.probs,
                logits=other.logits,
                name=name or other.name,
            )
            return result.with_source(
                Provenance("from_distribution", parents=(other,))
            )
        total_count = kwargs.pop("total_count", None)
        if total_count is None:
            raise ValueError(
                "total_count must be provided as a keyword argument "
                "when converting a non-Binomial distribution to Binomial."
            )
        num_samples = kwargs.pop("num_samples", 1024)
        total_count = jnp.asarray(total_count, dtype=jnp.float32)
        probs = other.mean() / total_count
        result = cls(total_count=total_count, probs=probs, name=name or other.name)
        return result.with_source(
            Provenance("from_distribution", parents=(other,))
        )


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

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Poisson:
        if isinstance(other, Poisson):
            result = cls(
                rate=other.rate,
                name=name or other.name,
            )
            return result.with_source(
                Provenance("from_distribution", parents=(other,))
            )
        num_samples = kwargs.pop("num_samples", 1024)
        rate = other.mean()
        result = cls(rate=rate, name=name or other.name)
        return result.with_source(
            Provenance("from_distribution", parents=(other,))
        )


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

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Categorical:
        if isinstance(other, Categorical):
            result = cls(
                probs=other.probs,
                logits=other.logits,
                name=name or other.name,
            )
            return result.with_source(
                Provenance("from_distribution", parents=(other,))
            )
        num_samples = kwargs.pop("num_samples", 1024)
        samples = other.sample(key, sample_shape=(num_samples,))
        num_categories = int(jnp.max(samples)) + 1
        counts = jnp.zeros(num_categories)
        for i in range(num_categories):
            counts = counts.at[i].set(jnp.sum(samples == i))
        probs = counts / jnp.sum(counts)
        result = cls(probs=probs, name=name or other.name)
        return result.with_source(
            Provenance("from_distribution", parents=(other,))
        )


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

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> NegativeBinomial:
        """Fit a NegativeBinomial from *other*.

        Keyword Args
        -------------
        total_count : int
            Number of failures. **Required** unless *other* is already a
            ``NegativeBinomial`` (in which case its ``total_count`` is
            reused).
        num_samples : int
            Samples drawn for moment-matching (default 1024).
        """
        if isinstance(other, NegativeBinomial):
            result = cls(
                total_count=other.total_count,
                probs=other.probs,
                logits=other.logits,
                name=name or other.name,
            )
            return result.with_source(
                Provenance("from_distribution", parents=(other,))
            )
        total_count = kwargs.pop("total_count", None)
        if total_count is None:
            raise ValueError(
                "total_count must be provided as a keyword argument "
                "when converting a non-NegativeBinomial distribution "
                "to NegativeBinomial."
            )
        num_samples = kwargs.pop("num_samples", 1024)
        total_count = jnp.asarray(total_count, dtype=jnp.float32)
        mean = other.mean()
        probs = total_count / (total_count + mean)
        result = cls(
            total_count=total_count, probs=probs, name=name or other.name
        )
        return result.with_source(
            Provenance("from_distribution", parents=(other,))
        )
