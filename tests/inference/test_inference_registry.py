"""Tests for the inference method registry."""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    GLMLikelihood,
    MultivariateNormal,
    Normal,
    ProductDistribution,
    SimpleModel,
    condition_on,
    mean,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.inference import inference_method_registry
from probpipe.modeling._likelihood import Likelihood

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_model():
    """A simple Poisson regression model."""
    import tensorflow_probability.substrates.jax.glm as tfp_glm

    X = np.asarray(np.linspace(-1, 1, 20))[:, None].astype(np.float32)
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
    return SimpleModel(prior, GLMLikelihood(tfp_glm.Poisson(), X))


@pytest.fixture
def data():
    return jnp.ones(20, dtype=float)


class TestInferenceMethodRegistry:
    def test_methods_registered(self):
        methods = inference_method_registry.list_methods()
        assert "tfp_nuts" in methods
        assert "tfp_hmc" in methods
        assert "blackjax_rwmh" in methods

    def test_priority_order(self):
        """Ranked methods precede the opt-in-only ones in the listing.

        ``tfp_nuts`` and ``tfp_hmc`` carry no priority, so they are listed
        but never selected automatically; ``blackjax_rwmh`` (55) is the
        gradient-free entry point automatic selection does reach.
        """
        methods = inference_method_registry.list_methods()
        assert {"tfp_nuts", "tfp_hmc", "blackjax_rwmh"}.issubset(methods)
        assert methods.index("blackjax_nuts") < methods.index("blackjax_rwmh")
        for opt_in in ("tfp_nuts", "tfp_hmc"):
            assert methods.index("blackjax_rwmh") < methods.index(opt_in)
            assert inference_method_registry.get_method(opt_in).priority is None

    def test_auto_select_nuts(self, simple_model, data):
        """BlackJAX NUTS is the auto-dispatch winner for any JAX-traceable model."""
        info = inference_method_registry.check(simple_model, data)
        assert info.feasible
        assert info.method_name == "blackjax_nuts"

    def test_method_override(self, simple_model, data):
        """method= should override auto-selection."""
        posterior = condition_on.apply(
            simple_model,
            data,
            method="blackjax_rwmh",
            num_results=50,
            num_warmup=20,
            random_seed=0,
        )
        assert posterior.algorithm == "blackjax_rwmh"

    def test_condition_on_default(self, simple_model, data):
        """Default condition_on should work through the registry."""
        posterior = condition_on(
            simple_model,
            data,
            num_results=50,
            num_warmup=20,
            random_seed=0,
        )
        assert mean(posterior).shape == (2,)

    def test_exact_only_refuses_every_inference_method(self, simple_model, data):
        """Every registered method is approximate, so an exact-only call resolves to nothing."""
        with pytest.raises(ResolutionError):
            condition_on(simple_model, data, exact_only=True)
        with pytest.raises(ResolutionError):
            condition_on(simple_model, data, method="blackjax_nuts", exact_only=True)

    def test_nonexistent_method_raises(self, simple_model, data):
        with pytest.raises(ResolutionError, match="nonexistent"):
            condition_on(simple_model, data, method="nonexistent")

    def test_infeasible_method_raises(self):
        """Requesting a method that can't handle the dist raises ResolutionError."""
        with pytest.raises(ResolutionError):
            inference_method_registry.execute("not_a_distribution", None, method="tfp_nuts")

    def test_bare_log_prob_distribution(self):
        """A bare SupportsLogProb distribution can be conditioned via registry.

        This tests "conditioning on nothing" — the posterior equals the
        prior since no observed data is provided.  Verifies that the
        registry can handle a plain distribution (not a model) when an
        explicit method is requested.
        """
        prior = Normal(loc=0.0, scale=1.0, name="x")
        posterior = condition_on(
            prior,
            method="tfp_nuts",
            num_results=50,
            num_warmup=20,
            random_seed=0,
        )
        assert mean(posterior).ndim <= 1

    def test_set_priorities_changes_selection(self, simple_model, data):
        """set_priorities should change which method is auto-selected."""
        original = inference_method_registry.get_method("blackjax_rwmh").priority
        inference_method_registry.set_priorities(blackjax_rwmh=200)
        try:
            info = inference_method_registry.check(simple_model, data)
            assert info.method_name == "blackjax_rwmh"
        finally:
            inference_method_registry.set_priorities(blackjax_rwmh=original)


# ---------------------------------------------------------------------------
# Built-in priority anchors
# ---------------------------------------------------------------------------


class TestBuiltInRanks:
    """Every built-in inference method is approximate; ranks order them.

    Methods whose ``check()`` is identical to a higher-ranked sibling are
    opt-in-only, ``priority=None``: they can never win auto-dispatch and are
    reachable only via ``method=`` (``blackjax_hmc`` vs ``blackjax_nuts``;
    ``blackjax_sghmc`` vs ``blackjax_sgld``). ``pymc_advi`` is also opt-in:
    VI is a deliberate bias-for-speed tradeoff the user should choose
    explicitly. ``tfp_nuts`` / ``tfp_hmc`` are opt-in for bit-pattern
    regression.
    """

    EXPECTED_PRIORITIES: ClassVar[dict[str, int | None]] = {
        "nutpie_nuts": 88,
        "blackjax_nuts": 85,
        "cmdstan_nuts": 82,
        "pymc_nuts": 82,
        "blackjax_elliptical_slice": 75,
        "blackjax_rwmh": 55,
        "blackjax_sgld": 45,
        "pyabc_smcabc": 6,
        # Opt-in only: registered but excluded from auto-dispatch.
        "blackjax_hmc": None,
        "blackjax_sghmc": None,
        "pymc_advi": None,
        "tfp_nuts": None,
        "tfp_hmc": None,
    }

    def test_every_registered_method_is_approximate(self):
        for name in inference_method_registry.list_methods():
            assert inference_method_registry.get_method(name).exact is False, name

    def test_ranks_match_anchors(self):
        # Asserts on the registered (class-level) rank so the test stays
        # valid if another test runs ``set_priorities(...)`` and forgets to
        # clean up: the override sits on the registry, not on the class.
        registered = inference_method_registry.list_methods()
        assert set(registered) <= set(self.EXPECTED_PRIORITIES), set(registered) - set(
            self.EXPECTED_PRIORITIES
        )
        for name, expected in self.EXPECTED_PRIORITIES.items():
            if name not in registered:
                continue  # optional backend not installed
            actual = inference_method_registry.get_method(name).priority
            assert actual == expected, f"{name} priority is {actual}, expected {expected}"

    def test_opt_in_set_is_exact(self):
        """The opt-in-only methods are exactly these; a rank on any other fails."""
        registered = set(inference_method_registry.list_methods())
        opt_in = {n for n in registered if inference_method_registry.get_method(n).priority is None}
        expected = {n for n, p in self.EXPECTED_PRIORITIES.items() if p is None} & registered
        assert opt_in == expected


# ---------------------------------------------------------------------------
# MCMC against unnormalized log densities
# ---------------------------------------------------------------------------


class _UnnormalizedTarget:
    """Mixin: implements only ``_unnormalized_log_prob`` (no ``_log_prob``).

    Used in the tests below to confirm that MCMC inference dispatches on
    :class:`SupportsUnnormalizedLogProb`, which is the strictly weaker
    protocol that MCMC actually needs.
    """

    def _unnormalized_log_prob(self, value):
        # Standard normal up to an unknown additive constant. The missing
        # log normalizer is irrelevant for accept/reject.
        return -0.5 * jnp.sum(value**2)

    def _mean(self):
        return jnp.zeros(2)


class _NormalizedTarget:
    """Mixin: implements only ``_log_prob`` (relies on protocol default).

    The :class:`SupportsLogProb` protocol provides a default
    ``_unnormalized_log_prob`` that delegates to ``_log_prob``; this
    fixture exercises that default path through the inference layer.
    """

    def _log_prob(self, value):
        return -0.5 * jnp.sum(value**2) - jnp.log(2 * jnp.pi)

    def _mean(self):
        return jnp.zeros(2)


def _make_unnormalized_distribution():
    from probpipe.core._distribution_base import Distribution

    class UnnormalizedDist(_UnnormalizedTarget, Distribution):
        event_shape = (2,)

        def __init__(self):
            super().__init__(name="unnorm")

    return UnnormalizedDist()


def _make_normalized_distribution():
    from probpipe.core._distribution_base import Distribution
    from probpipe.core.protocols import SupportsLogProb

    class NormalizedDist(_NormalizedTarget, Distribution, SupportsLogProb):
        # Inheriting SupportsLogProb gives the default
        # _unnormalized_log_prob (delegating to _log_prob) for free.
        event_shape = (2,)

        def __init__(self):
            super().__init__(name="norm")

    return NormalizedDist()


class TestUnnormalizedLogProbInference:
    """MCMC accepts distributions with only ``SupportsUnnormalizedLogProb``."""

    def test_unnormalized_only_satisfies_protocol(self):
        from probpipe.core.protocols import (
            SupportsLogProb,
            SupportsUnnormalizedLogProb,
        )

        dist = _make_unnormalized_distribution()
        assert isinstance(dist, SupportsUnnormalizedLogProb)
        assert not isinstance(dist, SupportsLogProb)

    def test_auto_dispatch_to_nuts(self):
        """Auto-dispatch picks blackjax_nuts for unnormalized-only target."""
        dist = _make_unnormalized_distribution()
        info = inference_method_registry.check(dist, None)
        assert info.feasible
        assert info.method_name == "blackjax_nuts"

    def test_condition_on_unnormalized_runs_nuts(self):
        from probpipe import ApproximateDistribution

        dist = _make_unnormalized_distribution()
        posterior = condition_on(
            dist,
            num_results=200,
            num_warmup=100,
            random_seed=0,
        )
        assert isinstance(posterior, ApproximateDistribution)
        # Standard normal: posterior mean ~0, std ~1 (loose tolerance —
        # short chain, no thinning).
        draws = np.asarray(posterior.draws()).reshape(-1, 2)
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.4)
        np.testing.assert_allclose(draws.std(0), [1.0, 1.0], atol=0.4)

    def test_condition_on_unnormalized_runs_rwmh(self):
        from probpipe import ApproximateDistribution

        dist = _make_unnormalized_distribution()
        posterior = condition_on(
            dist,
            method="blackjax_rwmh",
            num_results=200,
            num_warmup=100,
            step_size=0.5,
            random_seed=0,
        )
        assert isinstance(posterior, ApproximateDistribution)

    def test_normalized_only_still_works_via_nuts(self):
        """SupportsLogProb-only dist still flows through unchanged.

        Guards against a regression where the swap to
        ``_unnormalized_log_prob`` accidentally breaks the protocol's
        default delegation.
        """
        from probpipe import ApproximateDistribution

        dist = _make_normalized_distribution()
        posterior = condition_on(
            dist,
            num_results=100,
            num_warmup=50,
            random_seed=0,
        )
        assert isinstance(posterior, ApproximateDistribution)

    def test_normalized_only_still_works_via_rwmh(self):
        from probpipe import ApproximateDistribution

        dist = _make_normalized_distribution()
        posterior = condition_on(
            dist,
            method="blackjax_rwmh",
            num_results=100,
            num_warmup=50,
            step_size=0.5,
            random_seed=0,
        )
        assert isinstance(posterior, ApproximateDistribution)

    def test_check_description_names_unnormalized_protocol(self):
        """When MCMC methods are infeasible, error string names the right protocol."""
        from probpipe.core._distribution_base import Distribution

        class NoDensityDist(Distribution):
            event_shape = (2,)

            def __init__(self):
                super().__init__(name="no_density")

        dist = NoDensityDist()
        for method in ("tfp_nuts", "tfp_hmc", "blackjax_rwmh"):
            m = inference_method_registry.get_method(method)
            info = m.check(dist, None)
            assert not info.feasible
            assert "SupportsUnnormalizedLogProb" in info.description, (
                f"{method}: description {info.description!r} should mention "
                f"SupportsUnnormalizedLogProb"
            )


# ---------------------------------------------------------------------------
# Auto-dispatch correctness: NUTS vs ESS tier ordering
# ---------------------------------------------------------------------------


class _GaussianMeanLikelihood(Likelihood):
    """JAX-traceable Gaussian likelihood: ``mu`` is the flat parameter."""

    def log_likelihood(self, params, data):
        mu = jnp.reshape(jnp.asarray(params), ())
        return jnp.sum(-0.5 * (jnp.asarray(data) - mu) ** 2)


@pytest.fixture
def gaussian_model():
    """Gaussian-prior, JAX-traceable SimpleModel.

    Both ``blackjax_nuts`` (needs a traceable joint) and
    ``blackjax_elliptical_slice`` (needs a Gaussian prior + traceable
    likelihood + data) pass ``check()`` on this target — so it is the
    canonical case for testing the 85-vs-75 tier ordering.
    """
    prior = ProductDistribution(mu=Normal(loc=0.0, scale=1.0, name="mu"))
    return SimpleModel(prior, _GaussianMeanLikelihood(), name="gauss")


@pytest.fixture
def gaussian_data():
    return jnp.array([1.0, -1.0, 0.5])


class TestNutsEssDispatch:
    """NUTS (85) outranks ESS (75) on a target where both are feasible."""

    def test_both_methods_feasible(self, gaussian_model, gaussian_data):
        # Sanity: the scenario is only meaningful if ESS *would* fire
        # were NUTS absent. Confirm both pass check() in isolation.
        nuts = inference_method_registry.get_method("blackjax_nuts")
        ess = inference_method_registry.get_method("blackjax_elliptical_slice")
        assert nuts.check(gaussian_model, gaussian_data).feasible
        assert ess.check(gaussian_model, gaussian_data).feasible

    def test_nuts_wins_auto_dispatch(self, gaussian_model, gaussian_data):
        # Task 6: NUTS@85 outranks ESS@75 for a Gaussian-prior traceable
        # model. No Stan/PyMC NUTS variant (88/82) is feasible here —
        # they require StanModel / PyMCModel — so blackjax_nuts is the
        # highest-priority feasible method.
        info = inference_method_registry.check(gaussian_model, gaussian_data)
        assert info.feasible
        assert info.method_name == "blackjax_nuts"

    def test_ess_wins_when_nuts_demoted(self, gaussian_model, gaussian_data):
        # Task 7: a natural "gradient methods decline, ESS feasible"
        # scenario cannot be constructed cleanly here. ESS requires a
        # *JAX-traceable likelihood*, while NUTS requires a *traceable
        # joint* = (Gaussian prior log-prob, always traceable) +
        # likelihood. Any likelihood that makes NUTS' joint non-traceable
        # makes ESS' likelihood non-traceable too, so ESS would decline
        # alongside NUTS. We therefore use set_priorities to demote NUTS
        # below ESS (mirroring test_set_priorities_changes_selection),
        # with try/finally restore. Demoting 85 -> 70 stays positive, so
        # no opt-in-only zero-crossing warning fires.
        original = inference_method_registry.get_method("blackjax_nuts").priority
        inference_method_registry.set_priorities(blackjax_nuts=70)
        try:
            info = inference_method_registry.check(gaussian_model, gaussian_data)
            assert info.feasible
            assert info.method_name == "blackjax_elliptical_slice"
        finally:
            inference_method_registry.set_priorities(blackjax_nuts=original)
