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
        # NUTS wins auto-dispatch for JAX-traceable SupportsLogProb;
        # HMC is opt-in-only (same check() as NUTS would make it
        # structurally unreachable in auto-dispatch).
        assert nuts.priority == 85
        assert hmc.priority == 0


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
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=400, num_warmup=400, num_chains=2, random_seed=0,
        )
        m = mean(posterior)
        np.testing.assert_allclose(float(m["a"].squeeze()), 1.0, atol=0.15)
        np.testing.assert_allclose(float(m["b"].squeeze()), -2.0, atol=0.15)

    def test_closed_form_gaussian_target(self):
        """Single-parameter conjugate Gaussian: closed-form posterior recovery.

        Prior ``N(0, 1)``, likelihood is ``sum_i log N(y_i; mu, 1)``
        with ``y = [1.0, 2.0, 3.0]``. The posterior is ``N(1.5, 0.25)``:
        prior precision ``1`` + likelihood precision ``n = 3`` ⇒
        posterior precision ``4`` (variance ``0.25``); posterior mean
        is the precision-weighted average ``sum(y) / 4 = 1.5``.
        Tolerances below check mean to ~3 σ_MC and variance to 10%.
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
        instead of running ``window_adaptation``. We confirm the code
        path runs end-to-end *and* that the user-supplied step size
        propagates verbatim into ``sample_stats`` (no adaptation to
        overwrite it).
        """
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=50, num_warmup=0, step_size=0.05, random_seed=0,
        )
        m = mean(posterior)
        assert jnp.isfinite(m["a"]).all()
        assert jnp.isfinite(m["b"]).all()

        # With no warmup, the kernel runs at exactly the user step size.
        step_size = posterior.inference_data["sample_stats"]["step_size"]
        np.testing.assert_allclose(np.asarray(step_size), 0.05)


class TestBlackJAXHmc:
    """End-to-end smoke + correctness checks for ``blackjax_hmc``."""

    def test_runs_end_to_end(self, small_model):
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_hmc",
            num_results=200, num_warmup=200, num_chains=1,
            step_size=0.05, num_integration_steps=10, random_seed=0,
        )
        m = mean(posterior)
        assert m["a"].shape == ()
        assert m["b"].shape == ()

    def test_closed_form_gaussian_target(self):
        """HMC analogue of the NUTS closed-form Gaussian recovery.

        Prior ``N(0, 1)``, likelihood ``sum_i log N(y_i; mu, 1)`` with
        ``y = [1.0, 2.0, 3.0]`` ⇒ posterior ``N(1.5, 0.25)`` (precision
        ``1 + 3 = 4``; mean ``sum(y) / 4 = 1.5``).

        Pinned to HMC with ``num_integration_steps=5``: with the
        window-adapted step size this gives a trajectory length short
        enough to keep the sampler in the well-mixed regime. Empirically
        (8 seeds) the posterior-mean estimate has SD ≈ 0.005 and the
        variance estimate stays within ~3% of analytic, so the bands
        below (mean atol ``0.05`` ≈ several MC σ; variance rtol ``0.10``)
        are conservative MC-noise tolerances — far tighter than the
        ``O(0.5)`` error a mis-specified posterior would produce.
        """
        prior = ProductDistribution(mu=Normal(loc=0.0, scale=1.0, name="mu"))
        model = SimpleModel(prior, _GaussianMeanLikelihood(), name="g")
        y = jnp.asarray([1.0, 2.0, 3.0])

        posterior = condition_on(
            model, y, method="blackjax_hmc",
            num_results=4000, num_warmup=2000, num_chains=2,
            step_size=0.1, num_integration_steps=5, random_seed=0,
        )

        post_mean = float(mean(posterior)["mu"].squeeze())
        post_var = float(variance(posterior)["mu"].squeeze())
        np.testing.assert_allclose(post_mean, 1.5, atol=0.05)
        np.testing.assert_allclose(post_var, 0.25, rtol=0.10)


class TestSampleStats:
    """The ``sample_stats`` auxiliary group is populated correctly.

    Guards the contract that :func:`build_mcmc_datatree` and ArviZ
    expect: every diagnostic is shaped ``(chain, draw)`` and the
    injected ``step_size`` (which BlackJAX does *not* carry on its
    per-step ``info`` objects) actually lands in the group.
    """

    def test_sample_stats_keys_shapes_and_ranges(self, small_model):
        num_chains, num_results = 2, 200
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=num_results, num_warmup=200,
            num_chains=num_chains, random_seed=0,
        )
        ds = posterior.inference_data["sample_stats"]

        # NUTS plumbs these four info fields plus the injected step_size.
        expected = {
            "step_size", "acceptance_rate", "is_divergent",
            "num_integration_steps", "energy",
        }
        assert expected.issubset(set(ds.data_vars))

        for key in expected:
            assert ds[key].dims == ("chain", "draw")
            assert ds[key].shape == (num_chains, num_results)

        # BlackJAX reports the mean Metropolis acceptance probability,
        # which is mathematically in [0, 1] but can read 1.0 + a float32
        # epsilon from the exp/clip summation — allow that slack.
        ar = np.asarray(ds["acceptance_rate"])
        assert np.all(ar >= 0.0) and np.all(ar <= 1.0 + 1e-5)

        div = np.asarray(ds["is_divergent"])
        assert div.dtype == np.bool_
        # A well-adapted NUTS run on a Gaussian prior should rarely diverge.
        assert div.mean() < 0.05

        # step_size is injected from the warmup (no per-step info field),
        # so it is constant within a chain and strictly positive.
        step_size = np.asarray(ds["step_size"])
        assert np.all(step_size > 0.0)
        for c in range(num_chains):
            np.testing.assert_allclose(step_size[c], step_size[c, 0])

    def test_posterior_has_one_chain_dim_per_chain(self, small_model):
        num_chains = 2
        posterior = condition_on(
            small_model, jnp.zeros((4,)), method="blackjax_nuts",
            num_results=100, num_warmup=100, num_chains=num_chains, random_seed=0,
        )
        post_grp = posterior.inference_data["posterior"]
        assert post_grp.sizes["chain"] == num_chains
        assert posterior.inference_data["sample_stats"].sizes["chain"] == num_chains


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
