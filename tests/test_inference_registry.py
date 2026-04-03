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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """A simple Poisson regression model."""
    import tensorflow_probability.substrates.jax.glm as tfp_glm
    X = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)]).astype(np.float32)
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2))
    return SimpleModel(prior, GLMLikelihood(tfp_glm.Poisson(), X))


@pytest.fixture
def data():
    return jnp.ones(20, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Generic MethodRegistry tests
# ---------------------------------------------------------------------------

class TestMethodRegistry:

    def test_register_and_list(self):
        class FakeMethod(Method):
            def __init__(self, n, p):
                self._name = n
                self._priority = p
            @property
            def name(self): return self._name
            def supported_types(self): return (object,)
            @property
            def priority(self): return self._priority
            def check(self, *a, **kw): return MethodInfo(feasible=True, method_name=self._name)
            def execute(self, *a, **kw): return self._name

        reg = MethodRegistry()
        reg.register(FakeMethod("low", 10))
        reg.register(FakeMethod("high", 100))
        reg.register(FakeMethod("mid", 50))

        assert reg.list_methods() == ["high", "mid", "low"]

    def test_duplicate_name_raises(self):
        class FakeMethod(Method):
            @property
            def name(self): return "dup"
            def supported_types(self): return (object,)
            def check(self, *a, **kw): return MethodInfo(feasible=True)
            def execute(self, *a, **kw): return None

        reg = MethodRegistry()
        reg.register(FakeMethod())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(FakeMethod())

    def test_get_method(self):
        class FakeMethod(Method):
            @property
            def name(self): return "test"
            def supported_types(self): return (object,)
            def check(self, *a, **kw): return MethodInfo(feasible=True)
            def execute(self, *a, **kw): return "result"

        reg = MethodRegistry()
        m = FakeMethod()
        reg.register(m)
        assert reg.get_method("test") is m

    def test_get_method_not_found(self):
        reg = MethodRegistry()
        with pytest.raises(KeyError, match="No method named"):
            reg.get_method("nonexistent")

    def test_execute_by_name(self):
        class FakeMethod(Method):
            @property
            def name(self): return "test"
            def supported_types(self): return (object,)
            def check(self, *a, **kw): return MethodInfo(feasible=True)
            def execute(self, *a, **kw): return 42

        reg = MethodRegistry()
        reg.register(FakeMethod())
        assert reg.execute("anything", method="test") == 42

    def test_execute_no_method_raises(self):
        reg = MethodRegistry()
        with pytest.raises(TypeError, match="No method registered"):
            reg.execute("anything")

    def test_set_priorities(self):
        class FakeMethod(Method):
            def __init__(self, n, p):
                self._name = n
                self._priority = p
            @property
            def name(self): return self._name
            def supported_types(self): return (object,)
            @property
            def priority(self): return self._priority
            def check(self, *a, **kw): return MethodInfo(feasible=True, method_name=self._name)
            def execute(self, *a, **kw): return self._name

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
        assert posterior.diagnostics.algorithm == "rwmh"

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
        """A bare SupportsLogProb distribution can be conditioned via registry."""
        prior = Normal(loc=0.0, scale=1.0)
        posterior = condition_on(
            prior, method="tfp_nuts",
            num_results=50, num_warmup=20, random_seed=0,
        )
        assert mean(posterior).ndim <= 1

    def test_set_priorities_changes_selection(self, simple_model, data):
        """set_priorities should change which method is auto-selected."""
        # Make RWMH highest priority
        inference_method_registry.set_priorities(tfp_rwmh=200)
        try:
            info = inference_method_registry.check(simple_model, data)
            assert info.method_name == "tfp_rwmh"
        finally:
            # Restore original priority
            inference_method_registry.set_priorities(tfp_rwmh=50)
