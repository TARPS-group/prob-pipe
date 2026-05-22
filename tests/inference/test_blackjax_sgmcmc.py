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
    _build_grad_estimator,
)
from probpipe.inference._minibatch import MinibatchedDistribution


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
    # No-intercept logistic regression: prior dims pair 1-to-1 with X columns.
    lik = GLMLikelihood(tfp_glm.Bernoulli(), x=X, fit_intercept=False)
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
        """SGLD sits below full-batch NUTS so it only fires when
        explicitly requested; SGHMC is opt-in only.

        SGLD is the auto-dispatchable SGMCMC method at priority 45 —
        below ``blackjax_nuts`` (85) so a routine ``condition_on(...)``
        doesn't accidentally pick a stochastic-gradient sampler. SGHMC
        has the same ``check()`` as SGLD and is therefore structurally
        unreachable in auto-dispatch; it's at the opt-in sentinel
        ``priority=0`` and reachable only via
        ``method="blackjax_sghmc"``.
        """
        get = lambda n: inference_method_registry.get_method(n).priority
        # SGLD below the auto-dispatch winner (BlackJAX NUTS) but
        # positive so a `method="blackjax_sgld"` request still reaches
        # it through the priority walk.
        assert get("blackjax_sgld") < get("blackjax_nuts")
        assert get("blackjax_sgld") > 0
        # SGHMC at the opt-in sentinel.
        assert get("blackjax_sghmc") == 0


# -- Gradient-estimator correctness -------------------------------------------


class TestGradEstimatorCorrectness:
    """Direct verification that `_build_grad_estimator(measure)(theta, key)`
    matches the gradient through the exact same minibatch — no MC noise,
    no convergence slack. Catches sign-flip / scale bugs deterministically.
    """

    def test_grad_matches_full_data_grad_on_same_minibatch(self, logistic_problem):
        model = logistic_problem["model"]
        data = logistic_problem["data"]
        measure = MinibatchedDistribution(
            model.prior, model.likelihood, data, batch_size=20,
        )
        grad_estimator = _build_grad_estimator(measure)
        theta = jnp.array([0.13, -0.21])
        key = jax.random.PRNGKey(99)

        # What the kernel will compute, for one minibatch draw:
        actual = grad_estimator(theta, key)

        # Independent reference: rebuild the unnormalized log-density from
        # the captured batch via the prior + per-datum components directly,
        # then take its grad. The math here doesn't go through
        # `_FixedMinibatchDistribution._unnormalized_log_prob` at all.
        inner = measure._draw_one(key)
        batch = inner.batch
        rescale_factor = inner.rescale_factor

        def manual_log_density(t):
            per_datum = jax.vmap(
                model.likelihood.per_datum_log_likelihood, in_axes=(None, 0),
            )(t, batch)
            return model.prior._log_prob(t) + rescale_factor * jnp.sum(per_datum)

        expected = jax.grad(manual_log_density)(theta)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


# -- Reproducibility ---------------------------------------------------------


class TestReproducibility:
    """`random_seed` is a load-bearing reproducibility contract for
    inference results.
    """

    def test_same_seed_produces_identical_chain(self, logistic_problem):
        kwargs = dict(
            batch_size=20, num_steps=50, num_warmup=10,
            step_size=1e-3, random_seed=123,
        )
        post1 = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"], **kwargs,
        )
        post2 = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"], **kwargs,
        )
        np.testing.assert_array_equal(post1.flat_samples, post2.flat_samples)

    def test_different_seeds_produce_different_chains(self, logistic_problem):
        kwargs = dict(
            batch_size=20, num_steps=50, num_warmup=10, step_size=1e-3,
        )
        post1 = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            random_seed=1, **kwargs,
        )
        post2 = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            random_seed=2, **kwargs,
        )
        # Chains should differ somewhere — not just identical
        assert not jnp.allclose(post1.flat_samples, post2.flat_samples)


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

    Tolerances tightened beyond the original audit pass:
    `atol=0.3` is roughly the SE of the chain mean for this problem
    at 5000 retained samples plus warmup. Each test also asserts (a)
    chain finiteness and (b) a lower bound on per-coordinate std so
    a stuck (non-mixing) chain can't pass.
    """

    def test_sgld_recovers_logistic_coefficients(self, logistic_problem):
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=40, num_steps=5000, num_warmup=1000,
            step_size=1e-3, random_seed=42,
        )
        assert post.flat_samples.shape == (5000, 2)
        assert jnp.all(jnp.isfinite(post.flat_samples))
        # Non-mixing guard: a stuck chain near init would have ~zero std.
        per_coord_std = np.asarray(jnp.std(post.flat_samples, axis=0))
        assert per_coord_std.min() > 0.05, (
            f"Chain looks stuck — per-coord std: {per_coord_std}"
        )
        sample_mean = np.asarray(jnp.mean(post.flat_samples, axis=0))
        true = np.asarray(logistic_problem["true_theta"])
        np.testing.assert_allclose(sample_mean, true, atol=0.3)

    def test_sghmc_recovers_logistic_coefficients(self, logistic_problem):
        post = BlackJAXSGHMCMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=40, num_steps=5000, num_warmup=1000,
            step_size=2e-3, num_integration_steps=4,
            alpha=0.05, beta=0.0, random_seed=42,
        )
        assert post.flat_samples.shape == (5000, 2)
        assert jnp.all(jnp.isfinite(post.flat_samples))
        per_coord_std = np.asarray(jnp.std(post.flat_samples, axis=0))
        assert per_coord_std.min() > 0.05, (
            f"Chain looks stuck — per-coord std: {per_coord_std}"
        )
        sample_mean = np.asarray(jnp.mean(post.flat_samples, axis=0))
        true = np.asarray(logistic_problem["true_theta"])
        np.testing.assert_allclose(sample_mean, true, atol=0.3)


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

    def test_chain_shape_is_num_steps_by_event_shape(self, logistic_problem):
        """`post.flat_samples` is `(num_steps, *event_shape)` for a single chain."""
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=20, num_steps=100, num_warmup=0,
            step_size=1e-3, random_seed=1,
        )
        assert post.flat_samples.shape == (100, logistic_problem["P"])

    def test_warmup_discards_initial_samples(self, logistic_problem):
        """``num_warmup=N`` drops the first N samples; ``num_steps`` retained."""
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=20, num_steps=300, num_warmup=700,
            step_size=1e-3, random_seed=3,
        )
        assert post.flat_samples.shape == (300, 2)

    def test_user_supplied_init_position(self, logistic_problem):
        """``init=`` overrides the prior-sampled default."""
        init = jnp.array([2.5, -1.5])
        post = BlackJAXSGLDMethod().execute(
            logistic_problem["model"], logistic_problem["data"],
            batch_size=20, num_steps=50, num_warmup=0,
            step_size=1e-4, random_seed=0, init=init,
        )
        # With a tiny step size, the very first retained sample should
        # sit close to `init` (it's at most one Langevin step away).
        first = np.asarray(post.flat_samples[0])
        np.testing.assert_allclose(first, np.asarray(init), atol=0.05)

    def test_with_replacement_kwarg_dispatches(self, logistic_problem):
        """``with_replacement=True`` threads through the registry path."""
        post = condition_on(
            logistic_problem["model"], logistic_problem["data"],
            method="blackjax_sgld",
            batch_size=20, num_steps=100, num_warmup=0,
            step_size=1e-3, random_seed=4,
            with_replacement=True,
        )
        # No exception + finite chain == kwarg accepted by execute().
        assert jnp.all(jnp.isfinite(post.flat_samples))
