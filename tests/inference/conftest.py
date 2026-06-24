"""Shared fixtures for inference-method validation (issue #301).

A conjugate Gaussian linear model with a *closed-form* posterior — the trusted
reference an inference method is validated against by the suite in
``test_validation_suite.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import GLMLikelihood, MultivariateNormal, SimpleModel, condition_on
from probpipe.custom_types import Array
from probpipe.inference._approximate_distribution import ApproximateDistribution
from probpipe.validation import Reference


@dataclass(frozen=True)
class ConjugateLinearModel:
    """A conjugate Gaussian linear model and its exact analytic posterior."""

    model: SimpleModel
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
    likelihood = GLMLikelihood(tfp_glm.Normal(), x=x, fit_intercept=False)
    model = SimpleModel(
        MultivariateNormal(loc=jnp.zeros(p), cov=tau2 * jnp.eye(p), name="beta"), likelihood
    )
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
    return condition_on(
        m.model,
        m.data,
        method="blackjax_nuts",
        num_results=3000,
        num_warmup=1500,
        num_chains=2,
        random_seed=0,
    )
