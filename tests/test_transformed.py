"""Tests for TransformedDistribution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe.distributions import (
    ArrayDistribution,
    TransformedDistribution,
    Normal,
    MultivariateNormal,
    EmpiricalDistribution,
)
from probpipe.core.distribution import (
    real,
    positive,
    unit_interval,
)
from probpipe import log_prob, mean, sample, variance


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# TFP-backed base (Normal → Exp → positive samples)
# ---------------------------------------------------------------------------


class TestTFPBase:
    def test_exp_samples_positive(self, key):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        samples = sample(td, key=key, sample_shape=(100,))
        assert jnp.all(samples > 0)

    def test_exp_event_shape(self):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        assert td.event_shape == ()

    def test_exp_batch_shape(self):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        assert td.batch_shape == ()

    def test_exp_sample_shape(self, key):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        s = sample(td, key=key, sample_shape=(5,))
        assert s.shape == (5,)

    def test_exp_log_prob_shape(self, key):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        s = sample(td, key=key, sample_shape=(5,))
        lp = log_prob(td, s)
        assert lp.shape == (5,)

    def test_exp_log_prob_finite(self, key):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        s = sample(td, key=key, sample_shape=(10,))
        lp = log_prob(td, s)
        assert jnp.all(jnp.isfinite(lp))

    def test_sigmoid_samples_in_unit_interval(self, key):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Sigmoid())
        samples = sample(td, key=key, sample_shape=(100,))
        assert jnp.all(samples >= 0) and jnp.all(samples <= 1)

    def test_softplus_samples_positive(self, key):
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Softplus())
        samples = sample(td, key=key, sample_shape=(100,))
        assert jnp.all(samples > 0)

    def test_multivariate_exp(self, key):
        base = MultivariateNormal(
            loc=jnp.zeros(3), cov=jnp.eye(3)
        )
        td = TransformedDistribution(base, tfb.Exp())
        samples = sample(td, key=key, sample_shape=(10,))
        assert samples.shape == (10, 3)
        assert jnp.all(samples > 0)

    def test_mean_delegates_to_tfp_when_available(self):
        """Shift bijector preserves tractable mean."""
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Shift(5.0))
        m = mean(td)
        assert jnp.isclose(m, 5.0, atol=0.01)

    def test_variance_delegates_to_tfp_when_available(self):
        """Scale bijector has tractable variance."""
        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Scale(2.0))
        v = variance(td)
        assert jnp.isclose(v, 4.0, atol=0.01)


# ---------------------------------------------------------------------------
# Non-TFP base (EmpiricalDistribution → bijector)
# ---------------------------------------------------------------------------


class TestNonTFPBase:
    def test_exp_on_empirical(self, key):
        samples = jax.random.normal(key, (50, 2))
        emp = EmpiricalDistribution(samples)
        td = TransformedDistribution(emp, tfb.Exp())
        s = sample(td, key=key, sample_shape=(10,))
        assert s.shape == (10, 2)
        assert jnp.all(s > 0)

    def test_log_prob_on_empirical(self, key):
        samples = jax.random.normal(key, (50, 2))
        emp = EmpiricalDistribution(samples)
        td = TransformedDistribution(emp, tfb.Exp())
        x = jnp.ones(2)
        lp = log_prob(td, x)
        assert jnp.isfinite(lp)

    def test_mean_mc_fallback_on_non_tfp(self, key):
        """Non-TFP base: mean falls back to MC via expectation."""
        samples = jax.random.normal(key, (50, 2))
        emp = EmpiricalDistribution(samples)
        td = TransformedDistribution(emp, tfb.Exp())
        m = mean(td)
        assert jnp.all(jnp.isfinite(m))


# ---------------------------------------------------------------------------
# Chained bijectors
# ---------------------------------------------------------------------------


class TestChainedBijectors:
    def test_chain_sample_shape(self, key):
        base = Normal(loc=0.0, scale=1.0)
        chain = tfb.Chain([tfb.Exp(), tfb.Shift(1.0), tfb.Scale(2.0)])
        td = TransformedDistribution(base, chain)
        s = sample(td, key=key, sample_shape=(10,))
        assert s.shape == (10,)
        # Exp is the outermost bijector → samples should be positive
        assert jnp.all(s > 0)

    def test_chain_log_prob_finite(self, key):
        base = Normal(loc=0.0, scale=1.0)
        chain = tfb.Chain([tfb.Exp(), tfb.Shift(1.0), tfb.Scale(2.0)])
        td = TransformedDistribution(base, chain)
        s = sample(td, key=key, sample_shape=(5,))
        lp = log_prob(td, s)
        assert jnp.all(jnp.isfinite(lp))


# ---------------------------------------------------------------------------
# Support property
# ---------------------------------------------------------------------------


class TestSupport:
    def test_exp_support(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Exp())
        assert td.support == positive

    def test_sigmoid_support(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Sigmoid())
        assert td.support == unit_interval

    def test_softplus_support(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Softplus())
        assert td.support == positive

    def test_chain_support_from_outermost(self):
        chain = tfb.Chain([tfb.Exp(), tfb.Shift(1.0)])
        td = TransformedDistribution(Normal(0.0, 1.0), chain)
        assert td.support == positive

    def test_unknown_bijector_falls_back_to_real(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Shift(1.0))
        assert td.support == real


# ---------------------------------------------------------------------------
# Name and repr
# ---------------------------------------------------------------------------


class TestNameAndRepr:
    def test_name_default_none(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Exp())
        assert td.name is None

    def test_name_set(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Exp(), name="log_normal")
        assert td.name == "log_normal"

    def test_repr_contains_class_names(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Exp())
        r = repr(td)
        assert "TransformedDistribution" in r
        assert "Normal" in r
        assert "Exp" in r

    def test_is_distribution(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Exp())
        assert isinstance(td, ArrayDistribution)

    def test_dtype(self):
        td = TransformedDistribution(Normal(0.0, 1.0), tfb.Exp())
        assert td.dtype == jnp.float32
