"""Tests for the inference method registry."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    MultivariateNormal, Normal, ProductDistribution, SimpleModel, GLMLikelihood,
    condition_on, mean,
)
from probpipe.core._registry import (
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from probpipe.inference import inference_method_registry
from probpipe.modeling._likelihood import Likelihood


# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------

class FakeMethod(UnaryDispatchMethod):
    """Configurable stub for registry tests."""

    def __init__(self, n="fake", p=0, feasible=True, result=None):
        self._name = n
        self._priority = p
        self._feasible = feasible
        self._result = result

    @property
    def name(self):
        return self._name

    def supported_types(self):
        return (object,)

    @property
    def priority(self):
        return self._priority

    def check(self, *a, **kw):
        return MethodInfo(feasible=self._feasible, method_name=self._name)

    def execute(self, *a, **kw):
        return self._result if self._result is not None else self._name


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


# ---------------------------------------------------------------------------
# Generic UnaryDispatchRegistry tests
# ---------------------------------------------------------------------------

class TestUnaryDispatchRegistry:

    def test_register_and_list(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("low", 10))
        reg.register(FakeMethod("high", 100))
        reg.register(FakeMethod("mid", 50))
        assert reg.list_methods() == ["high", "mid", "low"]

    def test_duplicate_name_raises(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(FakeMethod("dup"))

    def test_get_method(self):
        reg = UnaryDispatchRegistry()
        m = FakeMethod("test")
        reg.register(m)
        assert reg.get_method("test") is m

    def test_get_method_not_found(self):
        reg = UnaryDispatchRegistry()
        with pytest.raises(KeyError, match="No method named"):
            reg.get_method("nonexistent")

    def test_execute_by_name(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("test", result=42))
        assert reg.execute("anything", method="test") == 42

    def test_execute_no_method_raises(self):
        reg = UnaryDispatchRegistry()
        with pytest.raises(TypeError, match="No method registered"):
            reg.execute("anything")

    def test_set_priorities(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("a", 10))
        reg.register(FakeMethod("b", 100))
        assert reg.list_methods() == ["b", "a"]

        reg.set_priorities(a=200)
        assert reg.list_methods() == ["a", "b"]

    def test_set_priorities_unknown_raises(self):
        reg = UnaryDispatchRegistry()
        with pytest.raises(KeyError):
            reg.set_priorities(nonexistent=100)


# ---------------------------------------------------------------------------
# Inference method registry tests
# ---------------------------------------------------------------------------

class TestInferenceMethodRegistry:

    def test_methods_registered(self):
        methods = inference_method_registry.list_methods()
        assert "tfp_nuts" in methods
        assert "tfp_hmc" in methods
        assert "blackjax_rwmh" in methods

    def test_priority_order(self):
        """BlackJAX RWMH stays above the opt-in-only TFP gradient methods.

        After the BlackJAX migration, ``tfp_nuts`` and ``tfp_hmc`` are
        at priority 0 (opt-in only) and so don't participate in the
        order ``list_methods()`` exposes for auto-dispatch.
        ``blackjax_rwmh`` (priority 55) is the gradient-free
        auto-dispatch entry point.
        """
        methods = inference_method_registry.list_methods()
        # All three remain registered.
        assert {"tfp_nuts", "tfp_hmc", "blackjax_rwmh"}.issubset(methods)
        # blackjax_nuts (85) outranks blackjax_rwmh (55) outranks the opt-in TFP pair.
        assert methods.index("blackjax_nuts") < methods.index("blackjax_rwmh")

    def test_auto_select_nuts(self, simple_model, data):
        """BlackJAX NUTS is the auto-dispatch winner for any JAX-traceable model."""
        info = inference_method_registry.check(simple_model, data)
        assert info.feasible
        assert info.method_name == "blackjax_nuts"

    def test_method_override(self, simple_model, data):
        """method= should override auto-selection."""
        posterior = condition_on(
            simple_model, data, method="blackjax_rwmh",
            num_results=50, num_warmup=20, random_seed=0,
        )
        assert posterior.algorithm == "blackjax_rwmh"

    def test_condition_on_default(self, simple_model, data):
        """Default condition_on should work through the registry."""
        posterior = condition_on(
            simple_model, data,
            num_results=50, num_warmup=20, random_seed=0,
        )
        assert mean(posterior).shape == (2,)

    def test_nonexistent_method_raises(self, simple_model, data):
        with pytest.raises(KeyError):
            condition_on(simple_model, data, method="nonexistent")

    def test_infeasible_method_raises(self):
        """Requesting a method that can't handle the dist raises TypeError."""
        with pytest.raises(TypeError):
            inference_method_registry.execute(
                "not_a_distribution", None, method="tfp_nuts"
            )

    def test_bare_log_prob_distribution(self):
        """A bare SupportsLogProb distribution can be conditioned via registry.

        This tests "conditioning on nothing" — the posterior equals the
        prior since no observed data is provided.  Verifies that the
        registry can handle a plain distribution (not a model) when an
        explicit method is requested.
        """
        prior = Normal(loc=0.0, scale=1.0, name="x")
        posterior = condition_on(
            prior, method="tfp_nuts",
            num_results=50, num_warmup=20, random_seed=0,
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
# Opt-in-only sentinel (priority == 0)
# ---------------------------------------------------------------------------

class TestOptInOnlyPriority:
    """Priority 0 = opt-in only: skipped during auto-dispatch."""

    def test_priority_zero_skipped_in_auto_walk(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("opt_in", p=0, result=10))
        # Auto-dispatch finds no method because the only one is opt-in.
        with pytest.raises(TypeError, match="No method registered"):
            reg.execute("anything")

    def test_priority_zero_reachable_by_name(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("opt_in", p=0, result=10))
        # Explicit method= still works.
        assert reg.execute("anything", method="opt_in") == 10

    def test_default_priority_is_opt_in(self):
        """A UnaryDispatchMethod subclass without a priority override defaults to opt-in."""
        class Bare(UnaryDispatchMethod):
            @property
            def name(self):
                return "bare"
            def supported_types(self):
                return (object,)
            def check(self, *a, **kw):
                return MethodInfo(feasible=True, method_name="bare")
            def execute(self, *a, **kw):
                return "ran"
        reg = UnaryDispatchRegistry()
        reg.register(Bare())
        # Auto-dispatch skips it.
        with pytest.raises(TypeError):
            reg.execute("anything")
        # Explicit invocation finds it.
        assert reg.execute("anything", method="bare") == "ran"

    def test_priority_zero_alongside_positive(self):
        """A priority-0 method does not block a positive-priority method."""
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("opt_in", p=0, result="skipped"))
        reg.register(FakeMethod("auto", p=10, result="ran"))
        assert reg.execute("anything") == "ran"

    def test_promote_from_opt_in_via_set_priorities(self):
        """set_priorities can promote a priority-0 method into auto-dispatch."""
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("opt_in", p=0, result="ran"))
        # Suppress the crossing warning for this targeted check.
        with pytest.warns(UserWarning, match="out of opt-in-only"):
            reg.set_priorities(opt_in=10)
        assert reg.execute("anything") == "ran"


# ---------------------------------------------------------------------------
# set_priorities zero-crossing warning
# ---------------------------------------------------------------------------

class TestSetPrioritiesZeroCrossingWarning:

    def test_warn_when_demoting_to_opt_in(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("a", p=50))
        with pytest.warns(UserWarning, match="into opt-in-only"):
            reg.set_priorities(a=0)

    def test_warn_when_promoting_from_opt_in(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("a", p=0))
        with pytest.warns(UserWarning, match="out of opt-in-only"):
            reg.set_priorities(a=42)

    def test_no_warn_when_staying_positive(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("a", p=50))
        # Crossings of the 50 break are documentary; they should not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            reg.set_priorities(a=10)   # exact -> inexact
            reg.set_priorities(a=80)   # inexact -> exact

    def test_no_warn_when_staying_zero(self):
        reg = UnaryDispatchRegistry()
        reg.register(FakeMethod("a", p=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            reg.set_priorities(a=0)


# ---------------------------------------------------------------------------
# Built-in priority anchors (issue #189)
# ---------------------------------------------------------------------------

class TestBuiltInPriorityAnchors:
    """Priority anchors per issue #189, with the BlackJAX MCMC migration.

    Methods whose ``check()`` is identical to a higher-priority sibling
    are at the opt-in-only sentinel ``priority=0`` — they can never win
    auto-dispatch and are reachable only via ``method=`` (``blackjax_hmc``
    vs ``blackjax_nuts``; ``blackjax_sghmc`` vs ``blackjax_sgld``).
    ``pymc_advi`` is also opt-in: VI is a deliberate bias-for-speed
    tradeoff the user should choose explicitly. ``tfp_nuts`` /
    ``tfp_hmc`` are opt-in for bit-pattern regression.
    """

    EXPECTED_PRIORITIES = {
        "nutpie_nuts": 88,
        "blackjax_nuts": 85,
        "cmdstan_nuts": 82,
        "pymc_nuts": 82,
        "blackjax_elliptical_slice": 75,
        "blackjax_rwmh": 55,
        "blackjax_sgld": 45,
        # Opt-in only — registered but excluded from auto-dispatch.
        "blackjax_hmc": 0,
        "blackjax_sghmc": 0,
        "pymc_advi": 0,
        "tfp_nuts": 0,
        "tfp_hmc": 0,
    }

    def test_priorities_match_anchors(self):
        # Asserts on the *registered* (class-level) priority via
        # ``UnaryDispatchMethod.priority`` so the test stays valid even
        # if another test runs ``set_priorities(...)`` and forgets to
        # clean up: the override sits on the registry, not on the
        # class. To assert on the *effective* dispatch ordering instead,
        # use ``BaseDispatchRegistry._effective_priority(method)``.
        for name, expected in self.EXPECTED_PRIORITIES.items():
            if name not in inference_method_registry.list_methods():
                continue   # optional backend not installed
            actual = inference_method_registry.get_method(name).priority
            assert actual == expected, (
                f"{name} priority is {actual}, expected {expected}"
            )

    def test_exact_above_inexact(self):
        """Every exact-tier (>50) priority outranks every inexact-tier (<=50)."""
        registered = set(inference_method_registry.list_methods())
        exact_priorities = [
            p for n, p in self.EXPECTED_PRIORITIES.items()
            if n in registered and p > 50
        ]
        inexact_priorities = [
            p for n, p in self.EXPECTED_PRIORITIES.items()
            if n in registered and 0 < p <= 50
        ]
        if exact_priorities and inexact_priorities:
            assert min(exact_priorities) > max(inexact_priorities)


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
        return -0.5 * jnp.sum(value ** 2)

    def _mean(self):
        return jnp.zeros(2)


class _NormalizedTarget:
    """Mixin: implements only ``_log_prob`` (relies on protocol default).

    The :class:`SupportsLogProb` protocol provides a default
    ``_unnormalized_log_prob`` that delegates to ``_log_prob``; this
    fixture exercises that default path through the inference layer.
    """

    def _log_prob(self, value):
        return -0.5 * jnp.sum(value ** 2) - jnp.log(2 * jnp.pi)

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
            dist, num_results=200, num_warmup=100, random_seed=0,
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
            dist, method="blackjax_rwmh",
            num_results=200, num_warmup=100, step_size=0.5, random_seed=0,
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
            dist, num_results=100, num_warmup=50, random_seed=0,
        )
        assert isinstance(posterior, ApproximateDistribution)

    def test_normalized_only_still_works_via_rwmh(self):
        from probpipe import ApproximateDistribution

        dist = _make_normalized_distribution()
        posterior = condition_on(
            dist, method="blackjax_rwmh",
            num_results=100, num_warmup=50, step_size=0.5, random_seed=0,
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
