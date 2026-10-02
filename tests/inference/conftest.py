"""Shared fixtures for inference-method validation.

A conjugate Gaussian linear model with a *closed-form* posterior — the trusted
reference an inference method is validated against by the suite in
``test_validation_suite.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp

from probpipe import Beta, MultivariateNormal, NumericArraySpec, condition_on
from probpipe.core.constraints import boolean
from probpipe.custom_types import Array
from probpipe.distributions import Distribution
from probpipe.families import GaussianFamily, glm_likelihood
from probpipe.inference._approximate_distribution import ApproximateDistribution
from probpipe.validation import Reference
from tests.inference.canonical import ObservationKernel

tfd = tfp.distributions


@dataclass(frozen=True)
class ConjugateLinearModel:
    """A conjugate Gaussian linear model and its exact analytic posterior."""

    model: Distribution  # the joint of the response y and the coefficients beta
    design: Array  # X, shape (n, p)
    data: Array  # response y, shape (n,)
    reference: Reference  # exact posterior N(mN, SN)


@pytest.fixture(scope="session")
def conjugate_linear_model() -> ConjugateLinearModel:
    """``y ~ N(Xβ, 1)`` with prior ``β ~ N(0, τ²I)`` — posterior is exactly Gaussian.

    With unit noise the posterior is ``N(mN, SN)``, ``SN = (τ⁻²I + XᵀX)⁻¹``,
    ``mN = SN Xᵀy``, giving a closed-form reference to validate inference against.
    """
    n, p, tau2 = 60, 3, 4.0
    k_x, k_beta, k_y = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(k_x, (n, p))
    beta_star = jax.random.normal(k_beta, (p,)) * jnp.sqrt(tau2)
    y = x @ beta_star + jax.random.normal(k_y, (n,))
    likelihood = glm_likelihood("y", GaussianFamily(), X=x, dispersion=1.0)
    model = likelihood * MultivariateNormal(loc=jnp.zeros(p), cov=tau2 * jnp.eye(p), label="beta")
    cov = jnp.linalg.inv(jnp.eye(p) / tau2 + x.T @ x)
    mean = cov @ (x.T @ y)
    return ConjugateLinearModel(
        model=model, design=x, data=y, reference=Reference.from_moments(mean=mean, cov=cov)
    )


@pytest.fixture(scope="session")
def conjugate_nuts_posterior(
    conjugate_linear_model: ConjugateLinearModel,
) -> ApproximateDistribution:
    """A well-mixed NUTS fit of the conjugate model — the method under validation."""
    m = conjugate_linear_model
    return condition_on.with_options(
        method="blackjax_nuts",
        method_options={"num_results": 3000, "num_warmup": 1500, "num_chains": 2, "random_seed": 0},
    )(m.model, {"y": m.data})


@dataclass(frozen=True)
class BetaBernoulliModel:
    """A Beta-Bernoulli conjugate model with its exact (skewed) Beta posterior."""

    model: Distribution  # the joint of the responses y and the success probability theta
    data: Array  # 0/1 responses
    reference: Reference  # exact Beta(α+k, β+n−k) posterior — moments and draws
    posterior_skewness: float  # > 0 ⇒ the reference is genuinely non-Gaussian


@pytest.fixture(scope="session")
def beta_bernoulli_model() -> BetaBernoulliModel:
    """Beta(1, 1) prior + Bernoulli likelihood, k=2 of n=13 → posterior Beta(3, 12).

    A conjugate but markedly **non-Gaussian** target (skew ≈ 0.7, bounded on
    [0, 1], mode ≈ 0.15 so safely inside the boundaries) — chosen far enough from
    Gaussian that a *moment-matched* Gaussian is itself rejected by the
    distributional metrics (see the negative control in the suite), so validating
    inference here genuinely checks behavior beyond the Gaussian case. The data
    are fixed (deterministic success count) so the posterior never pins against
    0/1, and the exact posterior is sampled directly to give a draws reference.
    """
    alpha, beta, k, n = 1.0, 1.0, 2, 13
    data = jnp.concatenate([jnp.ones(k), jnp.zeros(n - k)])
    post_alpha, post_beta = alpha + k, beta + (n - k)
    mean = post_alpha / (post_alpha + post_beta)
    var = mean * (1.0 - mean) / (post_alpha + post_beta + 1.0)
    draws = jax.random.beta(jax.random.PRNGKey(99), post_alpha, post_beta, (5000,))[:, None]
    skew = float(jnp.mean((draws[:, 0] - mean) ** 3) / var**1.5)
    reference = Reference.from_moments(mean=jnp.array([mean]), cov=jnp.array([[var]]), draws=draws)
    # The success probability is the parameter, conjugate to the Beta prior (unlike
    # a logit-linear GLM), so the posterior is Beta in closed form.
    prior = Beta("theta", alpha, beta)
    likelihood = ObservationKernel(
        "y",
        {"theta": prior.event_spec.spec},
        NumericArraySpec((n,), jnp.float32, boolean),
        lambda theta: tfd.Independent(
            tfd.Bernoulli(probs=jnp.full(n, theta), dtype=jnp.float32), 1
        ),
    )
    model = likelihood * prior
    return BetaBernoulliModel(model=model, data=data, reference=reference, posterior_skewness=skew)


@pytest.fixture(scope="session")
def beta_bernoulli_nuts_posterior(
    beta_bernoulli_model: BetaBernoulliModel,
) -> ApproximateDistribution:
    """A NUTS fit of the constrained, skewed Beta-Bernoulli posterior."""
    m = beta_bernoulli_model
    return condition_on.with_options(
        method="blackjax_nuts",
        method_options={"num_results": 2000, "num_warmup": 1000, "num_chains": 2, "random_seed": 0},
    )(m.model, {"y": m.data})
