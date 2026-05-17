"""Tests for BlackJAX-backed SGMCMC methods (``blackjax_sgld`` / ``blackjax_sghmc``).

End-to-end coverage of the inference-method-registry path:
``condition_on(model, observed, method="blackjax_sgld", batch_size=…, …)``,
plus checks that the gradient estimator actually drives convergence
toward the posterior mode on a 200-row Bayesian logistic regression.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    ApproximateDistribution,
    GLMLikelihood,
    MultivariateNormal,
    Record,
    SimpleModel,
    condition_on,
    inference_method_registry,
)
from probpipe.inference._blackjax_sgmcmc import (
    BlackJAXSGHMCMethod,
    BlackJAXSGLDMethod,
)


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def logistic_problem():
    """200-row Bayesian logistic regression with 2 coefficients."""
    N, P = 200, 2
    true_theta = jnp.array([1.0, -0.5])
    X = jax.random.normal(jax.random.PRNGKey(0), (N, P))
    logits = X @ true_theta
    y = (
        jax.random.uniform(jax.random.PRNGKey(1), (N,))
        < jax.nn.sigmoid(logits)
    ).astype(jnp.float32)

    prior = MultivariateNormal(loc=jnp.zeros(P), cov=jnp.eye(P), name="theta")
    lik = GLMLikelihood(tfp_glm.Bernoulli(), x=X)
    model = SimpleModel(prior=prior, likelihood=lik)
    data = Record(X=X, y=y)
    return {
        "model": model, "data": data, "true_theta": true_theta,
        "N": N, "P": P,
    }


# -- Registry membership ------------------------------------------------------


class TestRegistry:
    def test_sgld_registered(self):
        assert "blackjax_sgld" in inference_method_registry.list_methods()

    def test_sghmc_registered(self):
        assert "blackjax_sghmc" in inference_method_registry.list_methods()

    def test_priorities_below_full_batch(self):
        """SGMCMC methods sit below full-batch gradient methods so they
        only fire when explicitly requested."""
        names = inference_method_registry.list_methods()
        get = lambda n: inference_method_registry.get_method(n).priority
        # SGLD/SGHMC below NUTS and HMC; SGHMC below SGLD.
        assert get("blackjax_sgld") < get("tfp_nuts")
        assert get("blackjax_sgld") < get("tfp_hmc")
        assert get("blackjax_sghmc") < get("tfp_nuts")
        assert get("blackjax_sghmc") < get("blackjax_sgld")
        # And both above zero (so an opt-in `method=...` reaches them).
        assert get("blackjax_sgld") > 0
        assert get("blackjax_sghmc") > 0


# -- Feasibility (check) ------------------------------------------------------


class TestCheck:
    def test_rejects_bare_supports_log_prob(self):
        """A non-SimpleModel target returns ``feasible=False`` with hint."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="x")
        info = BlackJAXSGLDMethod().check(prior, None, batch_size=10)
        assert not info.feasible
        assert "SimpleModel" in info.description

    def test_rejects_non_factorisable_likelihood(self):
        """SimpleModel + bare Likelihood (no per_datum) is rejected."""
        class _BareLikelihood:
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="x")
        model = SimpleModel(prior=prior, likelihood=_BareLikelihood())
        info = BlackJAXSGLDMethod().check(model, None, batch_size=10)
        assert not info.feasible
        assert "ConditionallyIndependentLikelihood" in info.description

    def test_requires_batch_size_kwarg(self, logistic_problem):
        """Missing ``batch_size=`` returns ``feasible=False`` with hint."""
        info = BlackJAXSGLDMethod().check(
            logistic_problem["model"], logistic_problem["data"],
        )
        assert not info.feasible
        assert "batch_size" in info.description

    def test_feasible_for_well_formed_input(self, logistic_problem):
        info = BlackJAXSGLDMethod().check(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=20,
        )
        assert info.feasible


# -- End-to-end SGMCMC convergence -------------------------------------------


class TestConvergence:
    """SGMCMC actually drives the chain toward the posterior mode.

    Tolerances are generous (atol ~ 0.5) because stochastic-gradient
    samplers have a step-size-vs-noise tradeoff and we don't run
    enough iterations for tight bounds; the point is to catch a
    catastrophic gradient-direction bug, not to grade convergence.
    """

    def test_sgld_recovers_logistic_coefficients(self, logistic_problem):
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=40, num_steps=2000, num_warmup=500,
            step_size=1e-3, random_seed=42,
        )
        # post.flat_samples shape: (num_steps, P)
        assert post.flat_samples.shape == (2000, 2)
        sample_mean = np.asarray(jnp.mean(post.flat_samples, axis=0))
        true = np.asarray(logistic_problem["true_theta"])
        np.testing.assert_allclose(sample_mean, true, atol=0.6)

    def test_sghmc_recovers_logistic_coefficients(self, logistic_problem):
        post = BlackJAXSGHMCMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=40, num_steps=2000, num_warmup=500,
            step_size=2e-3, num_integration_steps=4,
            alpha=0.05, beta=0.0, random_seed=42,
        )
        assert post.flat_samples.shape == (2000, 2)
        sample_mean = np.asarray(jnp.mean(post.flat_samples, axis=0))
        true = np.asarray(logistic_problem["true_theta"])
        np.testing.assert_allclose(sample_mean, true, atol=0.6)


# -- condition_on dispatch ---------------------------------------------------


class TestConditionOnDispatch:
    def test_sgld_via_condition_on(self, logistic_problem):
        post = condition_on(
            logistic_problem["model"], logistic_problem["data"],
            method="blackjax_sgld", batch_size=40,
            num_steps=1000, num_warmup=200, step_size=1e-3,
            random_seed=7,
        )
        assert isinstance(post, ApproximateDistribution)
        assert post.flat_samples.shape == (1000, 2)

    def test_chain_shape_matches_tfp_convention(self, logistic_problem):
        """Chain has the same ``(num_steps, *event_shape)`` layout as TFP MCMC."""
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=20, num_steps=100, num_warmup=0,
            step_size=1e-3, random_seed=1,
        )
        # Single chain → (1, 100, 2) before .flat_samples flattening.
        # .flat_samples is the (num_chains * num_steps, *event_shape) view.
        assert post.flat_samples.shape == (100, 2)

    def test_warmup_discards_initial_samples(self, logistic_problem):
        """``num_warmup=N`` drops the first N samples; ``num_steps`` retained."""
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=20, num_steps=300, num_warmup=700,
            step_size=1e-3, random_seed=3,
        )
        assert post.flat_samples.shape == (300, 2)
