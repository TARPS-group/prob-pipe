"""Tests for the BlackJAX-backed NUTS / HMC inference methods."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Normal,
    NumericArraySpec,
    condition_on,
    mean,
    variance,
    workflow_run,
)
from probpipe.distributions import FactoredDistribution
from probpipe.inference import inference_method_registry
from probpipe.inference._inference_utils import observed_target
from tests._posterior import arviz_data, flat_draws
from tests.inference._harness import validate_method
from tests.inference.canonical import ObservationKernel

# TFP/JAX emit a deprecation warning during random-key construction
# (``shape requires ndarray or scalar arguments, got <class 'NoneType'>``)
# that is unrelated to the code under test. Scope the suppression to that
# specific message so genuine deprecations elsewhere still surface.
pytestmark = pytest.mark.filterwarnings(
    "ignore:shape requires ndarray or scalar arguments:DeprecationWarning",
)


def _identity(prior, n=4):
    """``n`` observations whose density does not depend on the parameters, times *prior*.

    The posterior collapses to the prior.
    """
    return (
        ObservationKernel(
            "y",
            dict(prior.event_spec.components),
            NumericArraySpec((n,)),
            lambda **values: tfd.Independent(tfd.Normal(jnp.zeros(n), 1.0), 1),
        )
        * prior
    )


def _gaussian_mean(prior, n=3):
    """``y_i ~ N(mu, 1)`` for ``n`` observations, times *prior* over ``mu``.

    Prior ``N(0, 1)`` on ``mu`` paired with ``n`` observations gives
    posterior ``N(n * y_bar / (n + 1), 1 / (n + 1))``.
    """
    return (
        ObservationKernel(
            "y",
            dict(prior.event_spec.components),
            NumericArraySpec((n,)),
            lambda mu: tfd.Independent(tfd.Normal(jnp.broadcast_to(mu, (n,)), 1.0), 1),
        )
        * prior
    )


@pytest.fixture
def small_model():
    prior = Normal(loc=1.0, scale=0.5, label="a") * Normal(loc=-2.0, scale=0.7, label="b")
    return _identity(prior)


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
        assert hmc.priority is None


class TestBlackJAXNuts:
    """End-to-end smoke + correctness checks for ``blackjax_nuts``."""

    def test_runs_end_to_end(self, small_model):
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={
                    "num_results": 200,
                    "num_warmup": 200,
                    "num_chains": 2,
                },
            )(small_model, {"y": jnp.zeros((4,))})
        m = mean(posterior)
        assert m["mean(a)"].shape == ()
        assert m["mean(b)"].shape == ()

    def test_collapses_to_prior_under_identity_likelihood(self, small_model):
        # With an identity likelihood, the posterior is the prior.
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={
                    "num_results": 400,
                    "num_warmup": 400,
                    "num_chains": 2,
                },
            )(small_model, {"y": jnp.zeros((4,))})
        m = mean(posterior)
        # Observed across four workflow seeds: |mean error| 0.004-0.044.
        np.testing.assert_allclose(float(jnp.squeeze(m["mean(a)"])), 1.0, atol=0.15)
        np.testing.assert_allclose(float(jnp.squeeze(m["mean(b)"])), -2.0, atol=0.15)

    def test_closed_form_gaussian_target(self):
        """Single-parameter conjugate Gaussian: closed-form posterior recovery.

        Prior ``N(0, 1)``, likelihood is ``sum_i log N(y_i; mu, 1)``
        with ``y = [1.0, 2.0, 3.0]``. The posterior is ``N(1.5, 0.25)``:
        prior precision ``1`` + likelihood precision ``n = 3`` ⇒
        posterior precision ``4`` (variance ``0.25``); posterior mean
        is the precision-weighted average ``sum(y) / 4 = 1.5``.
        Tolerances below check mean to 6 σ_MC and variance to 15%.
        """
        prior = FactoredDistribution("prior", [Normal(loc=0.0, scale=1.0, label="mu")])
        model = _gaussian_mean(prior)
        y = jnp.asarray([1.0, 2.0, 3.0])

        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={
                    "num_results": 2000,
                    "num_warmup": 1000,
                    "num_chains": 2,
                },
            )(model, {"y": y})

        # Analytic posterior: N(1.5, 0.25). The MC std error of the mean of
        # 4000 independent draws is sqrt(0.25 / 4000) ≈ 0.0079 = σ_MC.
        # Observed across workflow seeds 0-15 and 3000: |mean error|
        # 0.002-0.032 (up to 4 σ_MC), variance error 0.2-7.2%.
        analytic_mean = 1.5
        analytic_var = 0.25
        sigma_mc = (analytic_var / (2 * 2000)) ** 0.5
        post_mean = float(jnp.squeeze(mean(posterior)["mean(mu)"]))
        post_var = float(jnp.squeeze(variance(posterior)["variance(mu)"]))
        np.testing.assert_allclose(post_mean, analytic_mean, atol=6 * sigma_mc)
        np.testing.assert_allclose(post_var, analytic_var, rtol=0.15)

    def test_zero_warmup_uses_user_step_size(self, small_model):
        """Exercises the ``_adapt`` fallback (``num_warmup == 0``).

        When the user explicitly sets ``num_warmup=0``, the runner
        builds the kernel directly from the supplied ``step_size``
        instead of running ``window_adaptation``. We confirm the code
        path runs end-to-end *and* that the user-supplied step size
        propagates verbatim into ``sample_stats`` (no adaptation to
        overwrite it).
        """
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={
                    "num_results": 50,
                    "num_warmup": 0,
                    "step_size": 0.05,
                },
            )(small_model, {"y": jnp.zeros((4,))})
        m = mean(posterior)
        assert jnp.isfinite(m["mean(a)"]).all()
        assert jnp.isfinite(m["mean(b)"]).all()

        # With no warmup, the kernel runs at exactly the user step size.
        step_size = arviz_data(posterior)["sample_stats"]["step_size"]
        np.testing.assert_allclose(np.asarray(step_size), 0.05)


class TestObservedDataInRawHosts:
    """Observed data held in a pandas or xarray object condition as their array does."""

    @pytest.mark.parametrize("host", ["pandas", "xarray"])
    def test_the_draws_match_those_of_the_array(self, host):
        y = np.array([0.4, 0.9, -0.1], dtype=np.float32)
        if host == "pandas":
            hosted = pytest.importorskip("pandas").Series(y, name="y")
        else:
            hosted = pytest.importorskip("xarray").DataArray(y, dims="observation")
        model = _gaussian_mean(Normal("mu", 0.0, 1.0))
        options = {"num_results": 50, "num_warmup": 50, "num_chains": 1}
        fit = condition_on.with_options(method="blackjax_nuts", method_options=options)
        with workflow_run(seed=0):
            from_host = flat_draws(fit(model, {"y": hosted}))
        with workflow_run(seed=0):
            from_array = flat_draws(fit(model, {"y": jnp.asarray(y)}))
        np.testing.assert_array_equal(from_host, from_array)


class TestBlackJAXHmc:
    """End-to-end smoke + correctness checks for ``blackjax_hmc``."""

    def test_runs_end_to_end(self, small_model):
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_hmc",
                method_options={
                    "num_results": 200,
                    "num_warmup": 200,
                    "num_chains": 1,
                    "step_size": 0.05,
                    "num_integration_steps": 10,
                },
            )(small_model, {"y": jnp.zeros((4,))})
        m = mean(posterior)
        assert m["mean(a)"].shape == ()
        assert m["mean(b)"].shape == ()

    def test_closed_form_gaussian_target(self):
        """HMC analogue of the NUTS closed-form Gaussian recovery.

        Prior ``N(0, 1)``, likelihood ``sum_i log N(y_i; mu, 1)`` with
        ``y = [1.0, 2.0, 3.0]`` ⇒ posterior ``N(1.5, 0.25)`` (precision
        ``1 + 3 = 4``; mean ``sum(y) / 4 = 1.5``).

        Pinned to HMC with ``num_integration_steps=5`` — now the *mean*
        trajectory length, since production randomizes the leapfrog count
        with a Halton sequence around this value. Observed across four
        workflow seeds, the posterior-mean error is 0.005-0.007 and the
        variance estimate stays within 0.4-3.4% of analytic, so the bands
        below (mean atol ``0.05`` ≈ several MC σ; variance rtol ``0.10``)
        are conservative MC-noise tolerances — far tighter than the
        ``O(0.5)`` error a mis-specified posterior would produce.
        """
        prior = FactoredDistribution("prior", [Normal(loc=0.0, scale=1.0, label="mu")])
        model = _gaussian_mean(prior)
        y = jnp.asarray([1.0, 2.0, 3.0])

        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_hmc",
                method_options={
                    "num_results": 4000,
                    "num_warmup": 2000,
                    "num_chains": 2,
                    "step_size": 0.1,
                    "num_integration_steps": 5,
                },
            )(model, {"y": y})

        post_mean = float(jnp.squeeze(mean(posterior)["mean(mu)"]))
        post_var = float(jnp.squeeze(variance(posterior)["variance(mu)"]))
        np.testing.assert_allclose(post_mean, 1.5, atol=0.05)
        np.testing.assert_allclose(post_var, 0.25, rtol=0.10)

    def test_trajectory_length_is_randomized(self):
        """Production HMC draws a *random* number of leapfrog steps.

        The ``num_integration_steps`` per-step diagnostic must take many
        distinct values (not a single constant, as fixed-``L`` HMC would)
        with mean close to the configured value. This is the direct,
        deterministic check that the Halton trajectory-length jitter is
        active.
        """
        prior = FactoredDistribution("prior", [Normal(loc=0.0, scale=1.0, label="mu")])
        model = _gaussian_mean(prior)
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_hmc",
                method_options={
                    "num_results": 1000,
                    "num_warmup": 500,
                    "num_chains": 2,
                    "step_size": 0.1,
                    "num_integration_steps": 10,
                },
            )(model, {"y": jnp.asarray([1.0, 2.0, 3.0])})
        steps = np.asarray(arviz_data(posterior)["sample_stats"]["num_integration_steps"])
        # Randomized, not a single fixed L.
        assert np.unique(steps).size >= 5
        # Mean trajectory length tracks the configured value.
        np.testing.assert_allclose(steps.mean(), 10, rtol=0.1)

    def test_default_num_integration_steps_recovers_variance(self):
        """The default mean ``num_integration_steps=10`` recovers the posterior.

        A *fixed* 10-step trajectory at the window-adapted step size can
        resonate on this near-Gaussian target and under-estimate the
        posterior variance by ~30% while still showing healthy acceptance
        and zero divergences (the fragility this change addresses).
        Randomizing the trajectory length around the same mean recovers
        the closed-form variance ``0.25`` to within a few percent — checked
        here at the default ``num_integration_steps`` rather than the
        hand-dodged value used above.
        """
        prior = FactoredDistribution("prior", [Normal(loc=0.0, scale=1.0, label="mu")])
        model = _gaussian_mean(prior)
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_hmc",
                method_options={
                    "num_results": 4000,
                    "num_warmup": 2000,
                    "num_chains": 2,
                    "step_size": 0.1,
                    "num_integration_steps": 10,
                },
            )(model, {"y": jnp.asarray([1.0, 2.0, 3.0])})
        post_mean = float(jnp.squeeze(mean(posterior)["mean(mu)"]))
        post_var = float(jnp.squeeze(variance(posterior)["variance(mu)"]))
        # Observed across four workflow seeds: |mean error| 0.001-0.004,
        # variance error 0.8-4.8%.
        np.testing.assert_allclose(post_mean, 1.5, atol=0.05)
        np.testing.assert_allclose(post_var, 0.25, rtol=0.12)

    def test_halton_steps_floored_at_one(self):
        """``_halton_steps_fn`` never returns a 0-step trajectory.

        BlackJAX's ``halton_trajectory_length`` spans ``[0, 2L-1]`` and
        returns ``0`` for a handful of counter values (a no-op leapfrog).
        The shared step-count helper floors at ``1``; checked over a wide
        sweep of counter values that includes indices which map to ``0``
        unfloored, so this deterministically exercises the clamp.
        """
        from probpipe.inference._blackjax_mcmc import _halton_steps_fn

        steps_fn = _halton_steps_fn(10)
        counts = np.asarray([int(steps_fn(jnp.asarray(i))) for i in range(5000)])
        assert counts.min() >= 1
        # Clamping the rare zeros leaves the mean essentially unchanged.
        np.testing.assert_allclose(counts.mean(), 10, rtol=0.05)

    def test_zero_warmup_runs_end_to_end(self):
        """HMC with ``num_warmup=0`` re-inits the production ``dynamic_hmc``
        from the fixed-``L`` ``HMCState`` position and runs.

        The existing zero-warmup test targets NUTS; this pins the HMC
        ``num_warmup == 0`` branch (no adapted step size / mass matrix —
        the user-supplied ``step_size`` is used directly) against the
        randomized-``L`` production kernel.
        """
        prior = FactoredDistribution("prior", [Normal(loc=0.0, scale=1.0, label="mu")])
        model = _gaussian_mean(prior)
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_hmc",
                method_options={
                    "num_results": 100,
                    "num_warmup": 0,
                    "num_chains": 1,
                    "step_size": 0.1,
                    "num_integration_steps": 10,
                },
            )(model, {"y": jnp.asarray([1.0, 2.0, 3.0])})
        draws = np.asarray(flat_draws(posterior)["mu"]).reshape(-1)
        assert draws.shape[0] == 100
        assert np.all(np.isfinite(draws))
        # Trajectory length is still randomized (and floored) on the
        # zero-warmup path.
        steps = np.asarray(arviz_data(posterior)["sample_stats"]["num_integration_steps"])
        assert steps.min() >= 1
        assert np.unique(steps).size >= 5


class TestSampleStats:
    """The ``sample_stats`` annotations group is populated correctly.

    Guards the contract that :func:`build_mcmc_datatree` and ArviZ
    expect: every diagnostic is shaped ``(chain, draw)`` and the
    injected ``step_size`` (which BlackJAX does *not* carry on its
    per-step ``info`` objects) actually lands in the group.
    """

    def test_sample_stats_keys_shapes_and_ranges(self, small_model):
        num_chains, num_results = 2, 200
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={
                    "num_results": num_results,
                    "num_warmup": 200,
                    "num_chains": num_chains,
                },
            )(small_model, {"y": jnp.zeros((4,))})
        ds = arviz_data(posterior)["sample_stats"]

        # NUTS plumbs these four info fields plus the injected step_size.
        expected = {
            "step_size",
            "acceptance_rate",
            "diverging",
            "num_integration_steps",
            "energy",
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

        div = np.asarray(ds["diverging"])
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
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={
                    "num_results": 100,
                    "num_warmup": 100,
                    "num_chains": num_chains,
                },
            )(small_model, {"y": jnp.zeros((4,))})
        post_grp = arviz_data(posterior)["posterior"]
        assert post_grp.sizes["chain"] == num_chains
        assert arviz_data(posterior)["sample_stats"].sizes["chain"] == num_chains

    def test_the_mcmc_diagnostics_count_the_divergences(self, small_model):
        """``add_mcmc_diagnostics`` records the sum of ``diverging`` over chains and draws."""
        from probpipe.diagnostics import add_mcmc_diagnostics

        posterior = condition_on.with_options(
            method="blackjax_nuts",
            method_options={"num_results": 100, "num_warmup": 100, "num_chains": 2},
        )(small_model, {"y": jnp.zeros((4,))})
        add_mcmc_diagnostics(posterior)
        expected = int(np.asarray(arviz_data(posterior)["sample_stats"]["diverging"]).sum())
        assert posterior.diagnostics.mcmc.n_divergences == expected


class TestCheckFeasibility:
    """``check()`` correctly rejects targets it can't run."""

    def test_check_rejects_non_logprob_target(self):
        method = inference_method_registry.get_method("blackjax_nuts")
        info = method.check("not a distribution")
        assert info.feasible is False
        assert "SupportsUnnormalizedLogProb" in info.description

    def test_check_passes_on_a_joint_at_its_data(self, small_model):
        method = inference_method_registry.get_method("blackjax_nuts")
        info = method.check(observed_target(small_model, {"y": jnp.zeros((4,))}))
        assert info.feasible is True


# ---------------------------------------------------------------------------
# The canonical cases of the cross-method validation harness
# ---------------------------------------------------------------------------

test_blackjax_nuts_canonical = validate_method("blackjax_nuts")
test_blackjax_hmc_canonical = validate_method("blackjax_hmc")
