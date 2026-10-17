"""Laws and kernels the correctness modules share, each with the closed form a test reads.

Provides:
  - ``ExactGaussianRegression``: the joint of a Gaussian prior and a linear
    Gaussian observation, which claims exact conditioning on the observation;
  - ``RecordObservationKernel``: an observation kernel whose given slots are
    records;
  - ``Population``, ``Groups``, and ``schools_likelihood``: the eight-schools
    model with its parameters as a nested record, ``population/mu``,
    ``population/tau``, and ``groups/theta_tilde``;
  - ``REAL`` and ``POSITIVE``: the scalar specs the laws declare.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import MultivariateNormal, NumericArraySpec, OutputSpec, Record, RecordSpec
from probpipe.core.constraints import positive, real
from probpipe.distributions import Distribution
from probpipe.distributions._capabilities import (
    SupportsExactConditioning,
    SupportsLogProb,
    SupportsSampling,
)
from probpipe.distributions._factored import _raw_record
from tests.inference import canonical
from tests.inference.canonical import (
    LeafReference,
    ObservationKernel,
    PosteriorReference,
    _Observations,
)

REAL = NumericArraySpec((), jnp.float32, real)
POSITIVE = NumericArraySpec((), jnp.float32, positive)


def _fields(value: Any) -> dict[str, Any]:
    """*value*, a record or a raw mapping, as the nested mapping of its raw leaves."""
    return _raw_record(value) if isinstance(value, Record) else dict(value)


# ---------------------------------------------------------------------------
# The exactly conditioned regression
# ---------------------------------------------------------------------------


class ExactGaussianRegression(
    Distribution, SupportsSampling, SupportsLogProb, SupportsExactConditioning
):
    """The joint of ``beta ~ N(0, v I)`` and ``y | beta ~ N(X beta, I)``, conditioned exactly.

    Its event exposes the record ``{beta, y}``, it samples ancestrally, and its
    density is the product of the two Gaussian densities. Conditioning on ``y``
    returns the closed-form posterior ``N(S Xᵀ y, S)`` with ``S = (I / v +
    Xᵀ X)⁻¹`` as a ``MultivariateNormal``, and its guard admits ``y`` alone.

    Parameters
    ----------
    X : Array
        The design, of shape ``(observations, features)``.
    prior_variance : float
        The prior variance ``v`` of each coefficient.
    """

    def __init__(self, X: Any, prior_variance: float) -> None:
        X = jnp.asarray(X, jnp.float32)
        n, p = X.shape
        super().__init__(
            RecordSpec(
                beta=NumericArraySpec((p,), jnp.float32, real),
                y=NumericArraySpec((n,), jnp.float32, real),
            ),
            label="regression",
        )
        self.X, self.prior_variance = X, float(prior_variance)

    def posterior_moments(self, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """The closed-form posterior mean and covariance at the observation *y*, in double precision."""
        X = np.asarray(self.X, np.float64)
        cov = np.linalg.inv(np.eye(X.shape[1]) / self.prior_variance + X.T @ X)
        return cov @ X.T @ np.asarray(y, np.float64), cov

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        k_beta, k_y = jax.random.split(key)
        n, p = self.X.shape
        beta = jnp.sqrt(self.prior_variance) * jax.random.normal(k_beta, (*sample_shape, p))
        y = beta @ self.X.T + jax.random.normal(k_y, (*sample_shape, n))
        return {"beta": beta.astype(jnp.float32), "y": y.astype(jnp.float32)}

    def _log_prob(self, value: Any) -> Any:
        fields = _fields(value)
        beta, y = jnp.asarray(fields["beta"]), jnp.asarray(fields["y"])
        prior = jax.scipy.stats.norm.logpdf(beta, 0.0, jnp.sqrt(self.prior_variance)).sum(-1)
        likelihood = jax.scipy.stats.norm.logpdf(y, beta @ self.X.T, 1.0).sum(-1)
        return prior + likelihood

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution:
        mean, cov = self.posterior_moments(_fields(given)["y"])
        return MultivariateNormal(
            "beta", jnp.asarray(mean, jnp.float32), cov=jnp.asarray(cov, jnp.float32)
        )

    def _condition_on_guard(self, paths: tuple[str, ...]) -> bool:
        """The closed form conditions on the observation ``y`` alone."""
        return tuple(paths) == ("y",)


def exact_reference(
    mean: np.ndarray, variance: np.ndarray, path: str = "beta"
) -> PosteriorReference:
    """The reference of a Gaussian posterior with these coordinate means and variances at *path*."""
    sd = np.sqrt(variance)
    leaf = LeafReference(
        mean=np.asarray(mean),
        variance=np.asarray(variance),
        quantiles={q: mean + sd * scipy.stats.norm.ppf(q) for q in canonical.INTERVAL_LEVELS},
    )
    return PosteriorReference({path: leaf}, "closed form")


# ---------------------------------------------------------------------------
# The eight-schools model over a nested parameter record
# ---------------------------------------------------------------------------

#: The record of the population parameters.
POPULATION = RecordSpec(mu=REAL, tau=POSITIVE)

#: The number of schools.
J = canonical.SCHOOL_EFFECTS.shape[0]

#: The record of the per-school parameters.
GROUPS = RecordSpec(theta_tilde=NumericArraySpec((J,), jnp.float32, real))


class Population(Distribution, SupportsSampling, SupportsLogProb):
    """The whole record term ``population``: ``mu ~ N(0, 5)`` and ``tau ~ HalfCauchy(0, 5)``."""

    def __init__(self) -> None:
        super().__init__(
            OutputSpec(population=POPULATION),
            label="population",
        )

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        k_mu, k_tau = jax.random.split(key)
        mu = canonical.MU_SCALE * jax.random.normal(k_mu, sample_shape)
        tau = canonical.TAU_SCALE * jnp.abs(jax.random.cauchy(k_tau, sample_shape))
        return {"mu": mu.astype(jnp.float32), "tau": tau.astype(jnp.float32)}

    def _log_prob(self, value: Any) -> Any:
        fields = _fields(value)
        mu, tau = jnp.asarray(fields["mu"]), jnp.asarray(fields["tau"])
        return tfd.Normal(0.0, canonical.MU_SCALE).log_prob(mu) + tfd.HalfCauchy(
            0.0, canonical.TAU_SCALE
        ).log_prob(tau)


class Groups(Distribution, SupportsSampling, SupportsLogProb):
    """The whole record term ``groups``: ``theta_tilde ~ N(0, I)``, one coordinate per school."""

    def __init__(self) -> None:
        super().__init__(
            OutputSpec(groups=GROUPS),
            label="groups",
        )

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return {"theta_tilde": jax.random.normal(key, (*sample_shape, J)).astype(jnp.float32)}

    def _log_prob(self, value: Any) -> Any:
        theta_tilde = jnp.asarray(_fields(value)["theta_tilde"])
        return jax.scipy.stats.norm.logpdf(theta_tilde).sum(-1)


def schools_likelihood() -> ObservationKernel:
    """The kernel ``y | population, groups ~ N(mu + tau theta_tilde, sigma)``, one school per entry."""
    sigma = jnp.asarray(canonical.SCHOOL_ERRORS, jnp.float32)

    def build(population: Any, groups: Any) -> tfd.Distribution:
        mu, tau = jnp.asarray(population["mu"]), jnp.asarray(population["tau"])
        location = mu + tau * jnp.asarray(groups["theta_tilde"])
        return tfd.Independent(tfd.Normal(loc=location, scale=sigma), 1)

    return RecordObservationKernel(
        "y",
        {"population": POPULATION, "groups": GROUPS},
        NumericArraySpec((J,), jnp.float32, real),
        build,
    )


class RecordObservationKernel(ObservationKernel):
    """An observation kernel whose given slots are records, read as nested mappings of raw leaves."""

    def _law(self, given: Any) -> Distribution:
        top = dict(given.children if isinstance(given, Record) else given)
        values = {slot: _fields(value) for slot, value in top.items()}
        return _Observations(
            self.label, self._build(**values), self._support, event_spec=self.event_spec
        )


def nested_schools() -> Distribution:
    """The eight-schools joint ``p(y | population, groups) p(groups) p(population)``."""
    return schools_likelihood() * Groups() * Population()


def nested_schools_reference() -> PosteriorReference:
    """The eight-schools reference at the nested paths of the parameter record."""
    leaves = canonical.case("eight_schools").reference.leaves
    return PosteriorReference(
        {
            "population/mu": leaves["mu"],
            "population/tau": leaves["tau"],
            "groups/theta_tilde": leaves["theta_tilde"],
        },
        "quadrature",
    )
