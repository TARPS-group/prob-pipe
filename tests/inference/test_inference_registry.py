"""Tests for the inference method registry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    MultivariateNormal, Normal, SimpleModel, GLMLikelihood,
    condition_on, mean,
)
from probpipe.core._registry import Method, MethodInfo, MethodRegistry
from probpipe.inference import inference_method_registry


# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------

class FakeMethod(Method):
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
    X = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)]).astype(np.float32)
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
    return SimpleModel(prior, GLMLikelihood(tfp_glm.Poisson(), X))


@pytest.fixture
def data():
    return jnp.ones(20, dtype=float)


# ---------------------------------------------------------------------------
# Generic MethodRegistry tests
# ---------------------------------------------------------------------------

class TestMethodRegistry:

    def test_register_and_list(self):
        reg = MethodRegistry()
        reg.register(FakeMethod("low", 10))
        reg.register(FakeMethod("high", 100))
        reg.register(FakeMethod("mid", 50))
        assert reg.list_methods() == ["high", "mid", "low"]

    def test_duplicate_name_raises(self):
        reg = MethodRegistry()
        reg.register(FakeMethod("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(FakeMethod("dup"))

    def test_get_method(self):
        reg = MethodRegistry()
        m = FakeMethod("test")
        reg.register(m)
        assert reg.get_method("test") is m

    def test_get_method_not_found(self):
        reg = MethodRegistry()
        with pytest.raises(KeyError, match="No method named"):
            reg.get_method("nonexistent")

    def test_execute_by_name(self):
        reg = MethodRegistry()
        reg.register(FakeMethod("test", result=42))
        assert reg.execute("anything", method="test") == 42

    def test_execute_no_method_raises(self):
        reg = MethodRegistry()
        with pytest.raises(TypeError, match="No method registered"):
            reg.execute("anything")

    def test_set_priorities(self):
        reg = MethodRegistry()
        reg.register(FakeMethod("a", 10))
        reg.register(FakeMethod("b", 100))
        assert reg.list_methods() == ["b", "a"]

        reg.set_priorities(a=200)
        assert reg.list_methods() == ["a", "b"]

    def test_set_priorities_unknown_raises(self):
        reg = MethodRegistry()
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
        assert "tfp_rwmh" in methods

    def test_priority_order(self):
        methods = inference_method_registry.list_methods()
        nuts_idx = methods.index("tfp_nuts")
        hmc_idx = methods.index("tfp_hmc")
        rwmh_idx = methods.index("tfp_rwmh")
        assert nuts_idx < hmc_idx < rwmh_idx

    def test_auto_select_nuts(self, simple_model, data):
        """NUTS should be auto-selected for a JAX-traceable model."""
        info = inference_method_registry.check(simple_model, data)
        assert info.feasible
        assert info.method_name == "tfp_nuts"

    def test_method_override(self, simple_model, data):
        """method= should override auto-selection."""
        posterior = condition_on(
            simple_model, data, method="tfp_rwmh",
            num_results=50, num_warmup=20, random_seed=0,
        )
        assert posterior.algorithm == "rwmh"

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
        inference_method_registry.set_priorities(tfp_rwmh=200)
        try:
            info = inference_method_registry.check(simple_model, data)
            assert info.method_name == "tfp_rwmh"
        finally:
            inference_method_registry.set_priorities(tfp_rwmh=50)


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
        """Auto-dispatch picks tfp_nuts for unnormalized-only target."""
        dist = _make_unnormalized_distribution()
        info = inference_method_registry.check(dist, None)
        assert info.feasible
        assert info.method_name == "tfp_nuts"

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
            dist, method="tfp_rwmh",
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
            dist, method="tfp_rwmh",
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
        for method in ("tfp_nuts", "tfp_hmc", "tfp_rwmh"):
            m = inference_method_registry.get_method(method)
            info = m.check(dist, None)
            assert not info.feasible
            assert "SupportsUnnormalizedLogProb" in info.description, (
                f"{method}: description {info.description!r} should mention "
                f"SupportsUnnormalizedLogProb"
            )
