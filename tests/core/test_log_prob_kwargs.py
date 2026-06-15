"""Tests for the keyword form of the log_prob-family ops (issue #228).

``log_prob`` / ``prob`` / ``unnormalized_log_prob`` accept either a
positional value or named field kwargs; the kwargs are packed into a
single draw of the distribution's value type via
:meth:`Distribution._pack_value` (single-field → bare value; multi-field
→ ``Record``). This file exercises the keyword form across the
distribution surface, the error paths, and the reserved-name construction
warning.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    Beta,
    GLMLikelihood,
    MultivariateNormal,
    Normal,
    ProductDistribution,
    Record,
    SimpleModel,
    log_prob,
    prob,
    unnormalized_log_prob,
)
from probpipe.distributions._joint_gaussian import JointGaussian


class TestKwargFormScalar:
    """Single-field distributions: a field kwarg packs to the bare value."""

    def test_normal(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(log_prob(d, x=1.5), log_prob(d, 1.5))

    def test_beta(self):
        d = Beta(2.0, 3.0, name="p")
        assert jnp.allclose(log_prob(d, p=0.4), log_prob(d, 0.4))

    def test_multivariate_normal_vector_event(self):
        d = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        v = jnp.array([0.1, 0.2, 0.3])
        assert jnp.allclose(log_prob(d, z=v), log_prob(d, v))


class TestKwargFormRecord:
    """Multi-field distributions: field kwargs pack to a Record."""

    def test_product_distribution(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Beta(2.0, 3.0, name="b"))
        assert jnp.allclose(log_prob(p, a=0.5, b=0.4), log_prob(p, Record(a=0.5, b=0.4)))

    def test_field_order_independent(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Normal(0.0, 1.0, name="b"))
        # kwargs given out of template order still pack correctly
        assert jnp.allclose(log_prob(p, b=1.0, a=0.5), log_prob(p, a=0.5, b=1.0))

    def test_joint_gaussian(self):
        jg = JointGaussian(mean=jnp.zeros(3), cov=jnp.eye(3), u=2, v=1)
        u, v = jnp.array([0.1, 0.2]), jnp.array([0.3])
        assert jnp.allclose(log_prob(jg, u=u, v=v), log_prob(jg, Record(u=u, v=v)))


class TestKwargFormSimpleModel:
    """The headline case: ``log_prob(model, intercept=..., slope=..., X=..., y=...)``."""

    @staticmethod
    def _glm_model():
        X = np.array([0.1, 0.5, -0.3], dtype=np.float32)
        y = np.array([1.0, 3.0, 0.0], dtype=np.float32)
        lik = GLMLikelihood(tfp_glm.Poisson(), X)
        prior = ProductDistribution(
            Normal(0.0, 1.0, name="intercept"), Normal(0.0, 1.0, name="slope")
        )
        return SimpleModel(prior, lik, name="m"), X, y

    def test_kwarg_matches_record_and_tuple(self):
        model, X, y = self._glm_model()
        kw = log_prob(model, intercept=0.3, slope=0.5, X=X, y=y)
        rec = log_prob(model, Record(intercept=0.3, slope=0.5, X=X, y=y))
        tup = model._log_prob((Record(intercept=0.3, slope=0.5), Record(X=X, y=y)))
        assert jnp.allclose(kw, rec)
        assert jnp.allclose(kw, tup)


class TestStanModelPackValue:
    """StanModel Tier 1: the keyword form takes a single ``parameters=`` array.

    bridgestan is not installed in CI, so ``_pack_value`` is exercised in
    isolation via ``object.__new__`` (per STYLE_GUIDE §8.4).
    """

    def _bare_stan(self):
        from probpipe.modeling._stan import StanModel
        m = object.__new__(StanModel)
        m._name = "stan"
        return m

    def test_pack_value_parameters(self):
        m = self._bare_stan()
        arr = jnp.array([0.1, 0.2, 0.3])
        assert jnp.allclose(m._pack_value(parameters=arr), arr)

    def test_pack_value_wrong_kwargs_raises(self):
        m = self._bare_stan()
        with pytest.raises(TypeError, match="parameters="):
            m._pack_value(theta=jnp.array([0.1]))


class TestKwargErrors:
    def test_missing_field(self):
        p = ProductDistribution(Normal(0.0, 1.0, name="a"), Normal(0.0, 1.0, name="b"))
        with pytest.raises(TypeError, match="missing"):
            log_prob(p, a=0.5)

    def test_extra_field(self):
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="unexpected"):
            log_prob(d, x=1.0, bogus=2.0)

    def test_positional_value_and_kwargs(self):
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="not both"):
            log_prob(d, 1.0, x=2.0)

    def test_neither_value_nor_kwargs(self):
        d = Normal(0.0, 1.0, name="x")
        with pytest.raises(TypeError, match="value is required"):
            log_prob(d)


class TestProbAndUnnormalizedKwargForm:
    def test_prob_kwarg(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(prob(d, x=0.0), prob(d, 0.0))

    def test_unnormalized_log_prob_kwarg(self):
        d = Normal(0.0, 1.0, name="x")
        assert jnp.allclose(
            unnormalized_log_prob(d, x=1.2), unnormalized_log_prob(d, 1.2)
        )


class TestReservedNameWarning:
    """A single-field distribution named after a reserved WF control param
    warns at construction (the cheap check sees the field == the name)."""

    @pytest.mark.parametrize("reserved", ["seed", "n_broadcast_samples", "include_inputs"])
    def test_reserved_single_field_name_warns(self, reserved):
        with pytest.warns(UserWarning, match="reserved"):
            Normal(0.0, 1.0, name=reserved)

    def test_ordinary_name_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Normal(0.0, 1.0, name="x")
        assert not any("reserved" in str(w.message).lower() for w in caught)
