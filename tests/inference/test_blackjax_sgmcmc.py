"""Tests for BlackJAX-backed SGMCMC methods (``blackjax_sgld`` / ``blackjax_sghmc``).

End-to-end coverage of the inference-method-registry path:
``condition_on.with_options(method="blackjax_sgld", method_options={"batch_size": …})``
applied to ``(likelihood * prior, {"y": y})``,
plus checks that the gradient estimator actually drives convergence
toward the posterior mode on a 200-row Bayesian logistic regression.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    EmpiricalDistribution,
    HalfNormal,
    MultivariateNormal,
    NumericArraySpec,
    condition_on,
    inference_method_registry,
    workflow_run,
)
from probpipe.families import BernoulliFamily, GaussianFamily, glm_likelihood
from probpipe.inference._blackjax_sgmcmc import (
    BlackJAXSGHMCMethod,
    BlackJAXSGLDMethod,
    _build_grad_estimator,
)
from probpipe.inference._inference_utils import observed_target
from probpipe.inference._minibatch import MinibatchedDistribution
from tests._posterior import flat_chains
from tests.inference._harness import validate_method
from tests.inference.canonical import ObservationKernel


def _draws(posterior: EmpiricalDistribution) -> jax.Array:
    """The posterior's draws across its chains, in the target's flat layout."""
    return jnp.concatenate(flat_chains(posterior))


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def logistic_problem():
    """200-row Bayesian logistic regression with 2 coefficients."""
    N, P = 200, 2
    true_theta = jnp.array([1.0, -0.5])
    X = jax.random.normal(jax.random.PRNGKey(0), (N, P))
    logits = X @ true_theta
    y = (jax.random.uniform(jax.random.PRNGKey(1), (N,)) < jax.nn.sigmoid(logits)).astype(
        jnp.float32
    )

    prior = MultivariateNormal("beta", loc=jnp.zeros(P), cov=jnp.eye(P))
    # No-intercept logistic regression: prior dims pair 1-to-1 with X columns.
    lik = glm_likelihood("y", BernoulliFamily(), X=X)
    return {
        "model": lik * prior,
        "prior": prior,
        "likelihood": lik,
        "X": X,
        "y": y,
        "data": {"y": y},
        "true_theta": true_theta,
        "N": N,
        "P": P,
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
        unreachable in auto-dispatch; it is opt-in only, ``priority=None``,
        and reachable only via ``method="blackjax_sghmc"``.
        """

        def get(n):
            return inference_method_registry.get_method(n).priority

        # SGLD below the auto-dispatch winner (BlackJAX NUTS) but ranked, so
        # it participates in the auto walk.
        assert get("blackjax_sgld") < get("blackjax_nuts")
        assert get("blackjax_sgld") is not None
        # SGHMC is opt-in only.
        assert get("blackjax_sghmc") is None


# -- Gradient-estimator correctness -------------------------------------------


class TestGradEstimatorCorrectness:
    """Direct verification that `_build_grad_estimator(measure)(theta, key)`
    matches the gradient through the exact same minibatch — no MC noise,
    no convergence slack. Catches sign-flip / scale bugs deterministically.
    """

    def test_grad_matches_full_data_grad_on_same_minibatch(self, logistic_problem):
        prior, X, y = logistic_problem["prior"], logistic_problem["X"], logistic_problem["y"]
        measure = MinibatchedDistribution(
            "measure",
            prior,
            logistic_problem["likelihood"],
            y,
            batch_size=20,
        )
        grad_estimator = _build_grad_estimator(measure)
        theta = jnp.array([0.13, -0.21])
        key = jax.random.PRNGKey(99)

        # What the kernel will compute, for one minibatch draw:
        actual = grad_estimator(theta, key)

        # Independent reference: rebuild the unnormalized log-density from
        # the captured rows via the prior and the Bernoulli log-density of the
        # logits directly, then take its grad. The math here doesn't go
        # through `_FixedMinibatchDistribution._unnormalized_log_prob` at all.
        inner = measure._draw_one(key)
        rows = inner.rows
        rescale_factor = inner.rescale_factor

        def manual_log_density(t):
            logits = X[rows] @ t
            per_datum = tfd.Bernoulli(logits=logits).log_prob(y[rows])
            return prior._log_prob(t) + rescale_factor * jnp.sum(per_datum)

        expected = jax.grad(manual_log_density)(theta)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


# -- Reproducibility ---------------------------------------------------------


class TestReproducibility:
    """The seed of the workflow scope reproduces an SG-MCMC run."""

    def test_same_seed_produces_identical_chain(self, logistic_problem):
        kwargs = dict(
            batch_size=20,
            num_results=50,
            num_warmup=10,
            step_size=1e-3,
        )
        with workflow_run(seed=123):
            post1 = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                **kwargs,
            )
        with workflow_run(seed=123):
            post2 = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                **kwargs,
            )
        np.testing.assert_array_equal(_draws(post1), _draws(post2))

    def test_different_seeds_produce_different_chains(self, logistic_problem):
        kwargs = dict(
            batch_size=20,
            num_results=50,
            num_warmup=10,
            step_size=1e-3,
        )
        with workflow_run(seed=1):
            post1 = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                **kwargs,
            )
        with workflow_run(seed=2):
            post2 = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                **kwargs,
            )
        # Chains should differ somewhere — not just identical
        assert not jnp.allclose(_draws(post1), _draws(post2))


# -- Feasibility (check) ------------------------------------------------------


class TestCheck:
    def test_rejects_bare_supports_log_prob(self):
        """A target that is no factored joint at data returns ``feasible=False`` with hint."""
        prior = MultivariateNormal("x", loc=jnp.zeros(2), cov=jnp.eye(2))
        info = BlackJAXSGLDMethod().check(prior, batch_size=10)
        assert not info.feasible
        assert "likelihood * prior" in info.description

    def test_rejects_a_likelihood_that_scores_no_subset(self):
        """A likelihood kernel that cannot score a subset of its observations is rejected."""
        prior = MultivariateNormal("x", loc=jnp.zeros(2), cov=jnp.eye(2))
        likelihood = ObservationKernel(
            "y",
            {"x": prior.event_spec.spec},
            NumericArraySpec((5, 2)),
            lambda x: tfd.Independent(tfd.Normal(jnp.broadcast_to(x, (5, 2)), 1.0), 2),
        )
        target = observed_target(likelihood * prior, {"y": jnp.zeros((5, 2))})
        info = BlackJAXSGLDMethod().check(target, batch_size=2)
        assert not info.feasible
        assert "score a subset of its observations" in info.description

    def test_requires_batch_size_kwarg(self, logistic_problem):
        """Missing ``batch_size=`` returns ``feasible=False`` with hint."""
        info = BlackJAXSGLDMethod().check(
            observed_target(logistic_problem["model"], logistic_problem["data"]),
        )
        assert not info.feasible
        assert 'method_options={"batch_size": ...}' in info.description
        assert info.actionable

    def test_feasible_for_well_formed_input(self, logistic_problem):
        info = BlackJAXSGLDMethod().check(
            observed_target(logistic_problem["model"], logistic_problem["data"]),
            batch_size=20,
        )
        assert info.feasible


# -- End-to-end SGMCMC convergence -------------------------------------------


class TestConvergence:
    """SGMCMC actually drives the chain toward the posterior mode.

    Each test bounds the distance of the chain mean from the coefficients
    that generated the data by the spread it measured across workflow
    seeds, and also asserts (a) chain finiteness and (b) a lower bound on
    per-coordinate std so a stuck (non-mixing) chain can't pass.
    """

    def test_sgld_recovers_logistic_coefficients(self, logistic_problem):
        with workflow_run(seed=42):
            post = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                batch_size=40,
                num_results=5000,
                num_warmup=1000,
                step_size=1e-3,
            )
        assert _draws(post).shape == (5000, 2)
        assert jnp.all(jnp.isfinite(_draws(post)))
        # Non-mixing guard: a stuck chain near init would have ~zero std.
        per_coord_std = np.asarray(jnp.std(_draws(post), axis=0))
        # Observed across four workflow seeds: min std 0.16-0.17, and max
        # |mean - truth| 0.03-0.07.
        assert per_coord_std.min() > 0.05, f"Chain looks stuck — per-coord std: {per_coord_std}"
        sample_mean = np.asarray(jnp.mean(_draws(post), axis=0))
        true = np.asarray(logistic_problem["true_theta"])
        np.testing.assert_allclose(sample_mean, true, atol=0.3)

    def test_sghmc_recovers_logistic_coefficients(self, logistic_problem):
        with workflow_run(seed=42):
            post = BlackJAXSGHMCMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                batch_size=40,
                num_results=5000,
                num_warmup=1000,
                step_size=2e-3,
                num_integration_steps=4,
                alpha=0.05,
                beta=0.0,
            )
        assert _draws(post).shape == (5000, 2)
        assert jnp.all(jnp.isfinite(_draws(post)))
        per_coord_std = np.asarray(jnp.std(_draws(post), axis=0))
        # Observed across workflow seeds 40-51: min std 0.09-0.22, and max
        # |mean - truth| 0.08-0.44, since the chain mixes slowly at this step
        # size and friction.
        assert per_coord_std.min() > 0.05, f"Chain looks stuck — per-coord std: {per_coord_std}"
        sample_mean = np.asarray(jnp.mean(_draws(post), axis=0))
        true = np.asarray(logistic_problem["true_theta"])
        np.testing.assert_allclose(sample_mean, true, atol=0.8)


# -- condition_on dispatch ---------------------------------------------------


class TestConditionOnDispatch:
    def test_sgld_via_condition_on(self, logistic_problem):
        with workflow_run(seed=7):
            post = condition_on.with_options(
                method="blackjax_sgld",
                method_options={
                    "batch_size": 40,
                    "num_results": 1000,
                    "num_warmup": 200,
                    "step_size": 1e-3,
                },
            )(logistic_problem["model"], logistic_problem["data"])
        assert isinstance(post, EmpiricalDistribution)
        assert _draws(post).shape == (1000, 2)

    def test_chain_shape_is_num_results_by_event_shape(self, logistic_problem):
        """The draws are `(num_results, *event_shape)` for a single chain."""
        with workflow_run(seed=1):
            post = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                batch_size=20,
                num_results=100,
                num_warmup=0,
                step_size=1e-3,
            )
        assert _draws(post).shape == (100, logistic_problem["P"])

    def test_warmup_discards_initial_samples(self, logistic_problem):
        """``num_warmup=N`` drops the first N samples; ``num_results`` retained."""
        with workflow_run(seed=3):
            post = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                batch_size=20,
                num_results=300,
                num_warmup=700,
                step_size=1e-3,
            )
        assert _draws(post).shape == (300, 2)

    def test_user_supplied_init_position(self, logistic_problem):
        """``init=`` overrides the prior-sampled default."""
        init = jnp.array([2.5, -1.5])
        with workflow_run(seed=0):
            post = BlackJAXSGLDMethod().execute(
                observed_target(logistic_problem["model"], logistic_problem["data"]),
                batch_size=20,
                num_results=50,
                num_warmup=0,
                step_size=1e-4,
                init=init,
            )
        # With a tiny step size, the very first retained sample should
        # sit close to `init` (it's at most one Langevin step away, whose
        # noise has sd 0.014). Observed across four workflow seeds:
        # max |first - init| 0.008-0.025.
        first = np.asarray(_draws(post)[0])
        np.testing.assert_allclose(first, np.asarray(init), atol=0.05)

    def test_a_factored_prior_starts_the_chain_inside_its_support(self):
        """The chain over a factored prior starts at its draw, a positive dispersion."""
        n, p = 200, 2
        X = jax.random.normal(jax.random.PRNGKey(0), (n, p))
        y = X @ jnp.array([1.0, -0.5]) + 0.5 * jax.random.normal(jax.random.PRNGKey(1), (n,))
        prior = MultivariateNormal("beta", jnp.zeros(p), cov=jnp.eye(p)) * HalfNormal(
            "dispersion", 1.0
        )
        with workflow_run(seed=1):
            post = condition_on.with_options(
                method="blackjax_sgld",
                method_options={
                    "batch_size": 20,
                    "num_results": 50,
                    "num_warmup": 0,
                    "step_size": 1e-4,
                },
            )(glm_likelihood("y", GaussianFamily(), X=X) * prior, {"y": y})
        assert float(_draws(post)[0, 2]) > 0.0

    def test_with_replacement_kwarg_is_accepted_and_dispatches(self, logistic_problem):
        """``with_replacement=True`` is accepted and dispatches via the registry.

        This is a smoke test, not a behavioral one. ``with_replacement``
        only changes the index-draw inside
        :meth:`MinibatchedDistribution._draw_one` (``randint`` vs
        ``permutation``); it does not surface through the
        ``EmpiricalDistribution`` result, so there is no public handle
        that distinguishes a with- from a without-replacement run without
        contrived hooks into the minibatch RNG. We therefore assert only
        that the kwarg threads through ``condition_on`` -> ``execute()``
        and yields a finite, correctly-shaped chain. The replacement
        semantics themselves are covered directly in the
        ``MinibatchedDistribution`` tests.
        """
        with workflow_run(seed=4):
            post = condition_on.with_options(
                method="blackjax_sgld",
                method_options={
                    "batch_size": 20,
                    "num_results": 100,
                    "num_warmup": 0,
                    "step_size": 1e-3,
                    "with_replacement": True,
                },
            )(logistic_problem["model"], logistic_problem["data"])
        # No exception + finite, correctly-shaped chain == kwarg accepted
        # by execute() and threaded into MinibatchedDistribution.
        assert _draws(post).shape == (100, logistic_problem["P"])
        assert jnp.all(jnp.isfinite(_draws(post)))


# ---------------------------------------------------------------------------
# The canonical cases of the cross-method validation harness
# ---------------------------------------------------------------------------

test_blackjax_sgld_canonical = validate_method("blackjax_sgld")
test_blackjax_sghmc_canonical = validate_method("blackjax_sghmc")
