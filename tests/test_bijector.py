"""Tests for Bijector, TFPBijector, and pushforward via change-of-variables."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    Normal,
    MultivariateNormal,
    LogNormal,
    EmpiricalDistribution,
    Bijector,
    TFPBijector,
    BijectorTransformedDistribution,
    sample,
    log_prob,
)
from probpipe.core.distribution import (
    real,
    positive,
    unit_interval,
)
from probpipe.custom_types import Array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# TFPBijector wrapping
# ---------------------------------------------------------------------------


class TestTFPBijector:
    def test_forward(self):
        bij = TFPBijector(tfb.Exp())
        result = bij.forward(jnp.array(0.0))
        assert jnp.isclose(result, 1.0)

    def test_inverse(self):
        bij = TFPBijector(tfb.Exp())
        result = bij.inverse(jnp.array(1.0))
        assert jnp.isclose(result, 0.0)

    def test_call_delegates_to_forward(self):
        bij = TFPBijector(tfb.Exp())
        assert jnp.isclose(bij(0.0), 1.0)

    def test_forward_log_det_jacobian(self):
        bij = TFPBijector(tfb.Exp())
        # For exp(x) at x=0: |det J| = exp(0) = 1, log = 0
        ldj = bij.forward_log_det_jacobian(jnp.array(0.0))
        assert jnp.isclose(ldj, 0.0)

    def test_inverse_log_det_jacobian(self):
        bij = TFPBijector(tfb.Exp())
        # For log(y) at y=1: |det J| = 1/1 = 1, log = 0
        ldj = bij.inverse_log_det_jacobian(jnp.array(1.0))
        assert jnp.isclose(ldj, 0.0)

    def test_output_constraint_exp(self):
        bij = TFPBijector(tfb.Exp())
        assert bij.output_constraint == positive

    def test_output_constraint_sigmoid(self):
        bij = TFPBijector(tfb.Sigmoid())
        assert bij.output_constraint == unit_interval

    def test_output_constraint_softplus(self):
        bij = TFPBijector(tfb.Softplus())
        assert bij.output_constraint == positive

    def test_output_constraint_unknown(self):
        bij = TFPBijector(tfb.Shift(1.0))
        assert bij.output_constraint is None

    def test_output_constraint_chain_outermost(self):
        bij = TFPBijector(tfb.Chain([tfb.Exp(), tfb.Shift(1.0)]))
        assert bij.output_constraint == positive

    def test_tfp_bijector_accessor(self):
        raw = tfb.Exp()
        bij = TFPBijector(raw)
        assert bij.tfp_bijector is raw

    def test_repr(self):
        bij = TFPBijector(tfb.Exp())
        assert repr(bij) == "TFPBijector(Exp)"

    def test_isinstance_bijector(self):
        bij = TFPBijector(tfb.Exp())
        assert isinstance(bij, Bijector)


# ---------------------------------------------------------------------------
# Custom Bijector subclass (no TFP)
# ---------------------------------------------------------------------------


class ExpBijector(Bijector):
    """Pure-probpipe exp bijector for testing."""

    def forward(self, value):
        return jnp.exp(value)

    def inverse(self, value):
        return jnp.log(value)

    def forward_log_det_jacobian(self, value, event_ndims: int = 0):
        return value  # d/dx exp(x) = exp(x), log|det| = x

    @property
    def output_constraint(self):
        return positive


class TestCustomBijector:
    def test_forward(self):
        bij = ExpBijector()
        assert jnp.isclose(bij(0.0), 1.0)

    def test_inverse(self):
        bij = ExpBijector()
        assert jnp.isclose(bij.inverse(jnp.array(1.0)), 0.0)

    def test_forward_ldj(self):
        bij = ExpBijector()
        assert jnp.isclose(bij.forward_log_det_jacobian(jnp.array(0.0)), 0.0)

    def test_inverse_ldj_default(self):
        """Default inverse_log_det_jacobian = -forward_log_det_jacobian(inverse(y))."""
        bij = ExpBijector()
        y = jnp.array(1.0)
        ildj = bij.inverse_log_det_jacobian(y)
        # inverse(1.0) = 0.0, forward_ldj(0.0) = 0.0, so -0.0 = 0.0
        assert jnp.isclose(ildj, 0.0)

    def test_output_constraint(self):
        bij = ExpBijector()
        assert bij.output_constraint == positive


# ---------------------------------------------------------------------------
# Bijector.pushforward → BijectorTransformedDistribution
# ---------------------------------------------------------------------------


class TestBijectorPushforward:
    def test_exp_normal_returns_lognormal(self):
        """Exp + Normal dispatches to closed-form ExpNormalRule → LogNormal."""
        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(dist)
        assert isinstance(result, LogNormal)

    def test_cov_strategy_returns_bijector_transformed(self):
        """Explicit change_of_variables bypasses closed-form rule."""
        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(dist, strategy="change_of_variables")
        assert isinstance(result, BijectorTransformedDistribution)

    def test_sigmoid_returns_bijector_transformed(self):
        """Non-Exp bijector goes through CoV, not closed-form."""
        bij = TFPBijector(tfb.Sigmoid())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(dist)
        assert isinstance(result, BijectorTransformedDistribution)

    def test_samples_positive(self, key):
        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(dist)
        samples = sample(result, key=key, sample_shape=(100,))
        assert jnp.all(samples > 0)

    def test_log_prob_matches_lognormal(self, key):
        """Exp pushforward of N(0,1) should match LogNormal(0,1)."""
        bij = TFPBijector(tfb.Exp())
        base = Normal(loc=0.0, scale=1.0)
        transformed = bij.pushforward(base)
        reference = LogNormal(loc=0.0, scale=1.0)

        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        lp_transformed = log_prob(transformed, x)
        lp_reference = log_prob(reference, x)
        assert jnp.allclose(lp_transformed, lp_reference, atol=1e-5)

    def test_multivariate(self, key):
        bij = TFPBijector(tfb.Exp())
        base = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))
        result = bij.pushforward(base)
        # MultivariateNormal is not Normal, so ExpNormalRule doesn't fire → CoV
        assert isinstance(result, BijectorTransformedDistribution)
        samples = sample(result, key=key, sample_shape=(10,))
        assert samples.shape == (10, 3)
        assert jnp.all(samples > 0)

    def test_custom_bijector_pushforward(self, key):
        """Custom (non-TFP) bijector should produce BijectorTransformedDistribution."""
        bij = ExpBijector()
        base = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(base)
        assert isinstance(result, BijectorTransformedDistribution)
        samples = sample(result, key=key, sample_shape=(100,))
        assert jnp.all(samples > 0)

    def test_custom_bijector_log_prob(self, key):
        """Custom bijector log_prob should match reference LogNormal."""
        custom_bij = ExpBijector()
        base = Normal(loc=0.0, scale=1.0)
        custom_result = custom_bij.pushforward(base)
        reference = LogNormal(loc=0.0, scale=1.0)

        x = jnp.array([0.5, 1.0, 2.0])
        lp_custom = log_prob(custom_result, x)
        lp_ref = log_prob(reference, x)
        assert jnp.allclose(lp_custom, lp_ref, atol=1e-5)

    def test_provenance(self):
        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(dist)
        assert result.source is not None
        assert result.source.operation == "pushforward"
        assert result.source.parents == (dist,)

    def test_support_from_bijector(self):
        """CoV strategy for non-Normal → support from bijector."""
        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(dist, strategy="change_of_variables")
        assert result.support == positive

    def test_non_tfp_base(self, key):
        """Bijector on EmpiricalDistribution (non-TFP path)."""
        samples = jax.random.normal(key, (50, 2))
        emp = EmpiricalDistribution(samples)
        bij = TFPBijector(tfb.Exp())
        result = bij.pushforward(emp)
        assert isinstance(result, BijectorTransformedDistribution)
        s = sample(result, key=key, sample_shape=(10,))
        assert s.shape == (10, 2)
        assert jnp.all(s > 0)
