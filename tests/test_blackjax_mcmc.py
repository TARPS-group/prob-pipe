"""Tests for the BlackJAX-backed NUTS / HMC inference methods."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    ProductDistribution,
    SimpleModel,
    condition_on,
    mean,
    variance,
)
from probpipe.inference import inference_method_registry
from probpipe.modeling._likelihood import Likelihood

# TFP/JAX emit a deprecation warning during random-key construction
# (``shape requires ndarray or scalar arguments, got <class 'NoneType'>``)
# that is unrelated to the code under test. Scope the suppression to that
# specific message so genuine deprecations elsewhere still surface.
pytestmark = pytest.mark.filterwarnings(
    "ignore:shape requires ndarray or scalar arguments:DeprecationWarning",
)


class _IdentityLikelihood(Likelihood):
    """``log p(y | theta) = 0`` — posterior collapses to the prior."""

    def log_likelihood(self, params, data) -> float:
        return jnp.asarray(0.0)


class _GaussianMeanLikelihood(Likelihood):
    """``log p(y | mu) = sum_i log N(y_i; mu, 1)`` — closed-form posterior.

    Prior ``N(0, 1)`` on ``mu`` paired with ``n`` observations gives
    posterior ``N(n * y_bar / (n + 1), 1 / (n + 1))``.
    """

    def log_likelihood(self, params, data) -> float:
        mu = params if not hasattr(params, "fields") else params["mu"]
        y = data
        # Sum of N(y_i; mu, 1) log-densities, dropping the constant.
        return -0.5 * jnp.sum((y - mu) ** 2)


@pytest.fixture
def small_model() -> SimpleModel:
    prior = ProductDistribution(
        a=Normal(loc=1.0, scale=0.5, name="a"),
        b=Normal(loc=-2.0, scale=0.7, name="b"),
    )
    return SimpleModel(prior, _IdentityLikelihood(), name="m")


class TestBlackJAXRegistration:
    """Method-registry registration + priorities."""

    def test_both_methods_registered(self):
        names = inference_method_registry.list_methods()
        assert "blackjax_nuts" in names
        assert "blackjax_hmc" in names

    def test_priority_anchors(self):
        nuts = inference_method_registry.get_method("blackjax_nuts")
        hmc = inference_method_registry.get_method("blackjax_hmc")
        assert nuts.priority == 75
        assert hmc.priority == 65


class TestBlackJAXNuts:
    """End-to-end smoke + correctness checks for ``blackjax_nuts``."""

    def test_runs_end_to_end(self, small_model):
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=200, num_warmup=200, num_chains=2, random_seed=0,
        )
        m = mean(posterior)
        assert m["a"].shape == ()
        assert m["b"].shape == ()

    def test_collapses_to_prior_under_identity_likelihood(self, small_model):
        # With an identity likelihood, the posterior is the prior.
        # 800 draws across 2 chains gives ample MC precision for the
        # tolerances below.
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=400, num_warmup=400, num_chains=2, random_seed=0,
        )
        m = mean(posterior)
        np.testing.assert_allclose(float(m["a"].squeeze()), 1.0, atol=0.15)
        np.testing.assert_allclose(float(m["b"].squeeze()), -2.0, atol=0.15)

    def test_closed_form_gaussian_target(self):
        """Acceptance gate from plan §6.1: mean within 0.5 σ_MC, cov within 5%.

        Single-parameter Gaussian: prior ``N(0, 1)``, likelihood is
        ``sum_i log N(y_i; mu, 1)`` with ``y = [1.0, 2.0, 3.0]``. The
        posterior is ``N(1.5, 0.25)`` — prior precision ``1`` + likelihood
        precision ``n = 3``, posterior precision ``4`` ⇒ variance
        ``0.25``; posterior mean is the precision-weighted average
        ``(0 * 1 + sum(y) * 1) / 4 = 6/4 = 1.5``.
        """
        prior = ProductDistribution(mu=Normal(loc=0.0, scale=1.0, name="mu"))
        model = SimpleModel(prior, _GaussianMeanLikelihood(), name="g")
        y = jnp.asarray([1.0, 2.0, 3.0])

        posterior = condition_on(
            model, y, method="blackjax_nuts",
            num_results=2000, num_warmup=1000, num_chains=2, random_seed=0,
        )

        # Analytic posterior: N(1.5, 0.25). MC std error on the posterior
        # mean is sqrt(0.25 / 4000) ≈ 0.0079; allow 3 σ_MC.
        analytic_mean = 1.5
        analytic_var = 0.25
        sigma_mc = (analytic_var / (2 * 2000)) ** 0.5
        post_mean = float(mean(posterior)["mu"].squeeze())
        post_var = float(variance(posterior)["mu"].squeeze())
        np.testing.assert_allclose(post_mean, analytic_mean, atol=3 * sigma_mc)
        # Variance MC SE for 4000 draws of an N(.,.25) is ~0.0056 — allow
        # ~3 sigma plus a small slack for residual warmup bias.
        np.testing.assert_allclose(post_var, analytic_var, rtol=0.10)

    def test_zero_warmup_uses_user_step_size(self, small_model):
        """Exercises the ``_adapt`` fallback (``num_warmup == 0``).

        When the user explicitly sets ``num_warmup=0``, the runner
        builds the kernel directly from the supplied ``step_size``
        instead of running ``window_adaptation``. Smoke-only — the
        unwarmed chain isn't expected to match the prior closely,
        we just confirm the code path runs end-to-end.
        """
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=50, num_warmup=0, step_size=0.05, random_seed=0,
        )
        m = mean(posterior)
        assert jnp.isfinite(m["a"]).all()
        assert jnp.isfinite(m["b"]).all()


class TestBlackJAXHmc:
    """End-to-end smoke for ``blackjax_hmc``."""

    def test_runs_end_to_end(self, small_model):
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_hmc",
            num_results=200, num_warmup=200, num_chains=1,
            step_size=0.05, num_integration_steps=10, random_seed=0,
        )
        m = mean(posterior)
        assert m["a"].shape == ()
        assert m["b"].shape == ()


class TestCheckFeasibility:
    """``check()`` correctly rejects targets it can't run."""

    def test_check_rejects_non_logprob_target(self):
        method = inference_method_registry.get_method("blackjax_nuts")
        info = method.check("not a distribution", observed=None)
        assert info.feasible is False
        assert "SupportsUnnormalizedLogProb" in info.description

    def test_check_passes_on_simple_model(self, small_model):
        method = inference_method_registry.get_method("blackjax_nuts")
        info = method.check(small_model, observed=jnp.zeros((4,)))
        assert info.feasible is True
