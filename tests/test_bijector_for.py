"""Tests for Constraint → Bijector dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    Normal,
    TransformedDistribution,
    bijector_for,
    boolean,
    greater_than,
    integer_interval,
    interval,
    non_negative,
    non_negative_integer,
    positive,
    positive_definite,
    real,
    register_bijector,
    simplex,
    sphere,
    unit_interval,
    unregister_bijector,
)
from probpipe.core.constraints import Constraint


# ---------------------------------------------------------------------------
# Per-constraint round-trips
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_real_identity(self):
        bij = bijector_for(real)
        assert isinstance(bij, tfb.Identity)
        x = jnp.array([-3.0, 0.0, 4.5])
        assert jnp.allclose(bij.forward(x), x)
        assert jnp.allclose(bij.inverse(bij.forward(x)), x)

    def test_positive_exp(self):
        bij = bijector_for(positive)
        assert isinstance(bij, tfb.Exp)
        x = jnp.array([-2.0, 0.0, 3.0])
        y = bij.forward(x)
        assert jnp.all(positive.check(y))
        assert jnp.allclose(bij.inverse(y), x, atol=1e-5)

    def test_non_negative_softplus(self):
        bij = bijector_for(non_negative)
        assert isinstance(bij, tfb.Softplus)
        x = jnp.array([-100.0, -1.0, 0.0, 5.0])
        y = bij.forward(x)
        assert jnp.all(non_negative.check(y))

    def test_unit_interval_sigmoid(self):
        bij = bijector_for(unit_interval)
        assert isinstance(bij, tfb.Sigmoid)
        x = jnp.array([-100.0, 0.0, 100.0])
        y = bij.forward(x)
        assert jnp.all(unit_interval.check(y))

    def test_interval_sigmoid(self):
        c = interval(2.0, 5.0)
        bij = bijector_for(c)
        # Moderate scales — float32 saturates ``Sigmoid`` past ~|x|=15.
        x = jnp.array([-3.0, 0.0, 3.0])
        y = bij.forward(x)
        assert jnp.all(c.check(y))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(y > 2.0) and jnp.all(y < 5.0)
        # Round-trip.
        x0 = jnp.array([0.0, 1.0])
        assert jnp.allclose(bij.inverse(bij.forward(x0)), x0, atol=1e-5)

    def test_greater_than(self):
        c = greater_than(3.0)
        bij = bijector_for(c)
        x = jnp.array([-2.0, 0.0, 5.0])
        y = bij.forward(x)
        assert jnp.all(c.check(y))

    def test_simplex_softmax_centered(self):
        bij = bijector_for(simplex)
        assert isinstance(bij, tfb.SoftmaxCentered)
        x = jnp.array([0.5, -1.0])  # 2 unconstrained → 3-simplex
        y = bij.forward(x)
        assert simplex.check(y)
        assert y.shape == (3,)

    def test_positive_definite(self):
        bij = bijector_for(positive_definite)
        # FillScaleTriL: 6 unconstrained params → 3x3 lower triangular L;
        # CholeskyOuterProduct: L → L Lᵀ.
        x = jnp.array([1.0, 0.5, 2.0, -0.3, 0.1, 1.5])
        m = bij.forward(x)
        assert m.shape == (3, 3)
        assert positive_definite.check(m)
        # Symmetric.
        assert jnp.allclose(m, m.T, atol=1e-5)


# ---------------------------------------------------------------------------
# Vector / multivariate event shapes
# ---------------------------------------------------------------------------


class TestVectorEventShapes:
    def test_interval_with_vector_bounds(self):
        c = interval(jnp.zeros(3), jnp.array([1.0, 2.0, 3.0]))
        # Unhashable jax-array params: instance lookup must fall through
        # to type lookup cleanly (no TypeError surfacing).
        bij = bijector_for(c)
        x = jnp.array([-2.0, 0.0, 2.0])
        y = bij.forward(x)
        assert jnp.all(y >= 0.0)
        assert jnp.all(y <= jnp.array([1.0, 2.0, 3.0]))

    def test_positive_pointwise(self):
        bij = bijector_for(positive)
        x = jnp.array([[-1.0, 2.0], [3.0, -4.0]])
        y = bij.forward(x)
        assert jnp.all(y > 0)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Unsupported constraints
# ---------------------------------------------------------------------------


class TestUnsupported:
    @pytest.mark.parametrize(
        "constraint",
        [sphere, boolean, non_negative_integer, integer_interval(0, 5)],
    )
    def test_raises_not_implemented(self, constraint):
        with pytest.raises(NotImplementedError):
            bijector_for(constraint)

    def test_unregistered_custom_constraint_raises(self):
        class MyConstraint(Constraint):
            def check(self, value):
                return jnp.asarray(value) >= 0

        with pytest.raises(NotImplementedError, match="No bijector registered"):
            bijector_for(MyConstraint())


# ---------------------------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------------------------


class TestBoundary:
    """``Sigmoid`` and ``Exp`` saturate / overflow at extreme inputs in
    float32; ``bijector_for`` does not promise clipping.  These tests
    verify well-behaved output at scales typical of unconstrained
    optimization (roughly within ±10 in float32)."""

    def test_interval_moderate_inputs(self):
        c = interval(0.0, 1.0)
        bij = bijector_for(c)
        y = bij.forward(jnp.array([-5.0, 0.0, 5.0]))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all((y > 0.0) & (y < 1.0))

    def test_positive_moderate_inputs(self):
        bij = bijector_for(positive)
        y = bij.forward(jnp.array([-5.0, 0.0, 5.0]))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(y > 0)

    def test_greater_than_moderate_inputs(self):
        c = greater_than(-2.5)
        bij = bijector_for(c)
        y = bij.forward(jnp.array([-5.0, 5.0]))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(c.check(y))


# ---------------------------------------------------------------------------
# Integration with TransformedDistribution
# ---------------------------------------------------------------------------


class TestIntegration:
    """``TransformedDistribution.support`` reads the bijector class name
    from ``_BIJECTOR_SUPPORT_MAP`` in ``transformed.py``.  That map and
    ``bijector_for`` are not strict inverses: they only round-trip for the
    unparameterized bijectors that appear in the forward map."""

    def test_transformed_distribution_inherits_support_positive(self):
        base = Normal(loc=0.0, scale=1.0, name="x")
        td = TransformedDistribution(base, bijector_for(positive))
        assert td.support == positive

    def test_transformed_distribution_inherits_support_unit_interval(self):
        base = Normal(loc=0.0, scale=1.0, name="x")
        td = TransformedDistribution(base, bijector_for(unit_interval))
        assert td.support == unit_interval

    def test_round_trip_drifts_for_parameterized_interval(self):
        # ``bijector_for(interval(2,5))`` returns a parameterized
        # ``Sigmoid``; the forward map only knows the bijector class name
        # ``Sigmoid`` → ``unit_interval`` and cannot recover the bounds.
        base = Normal(loc=0.0, scale=1.0, name="x")
        td = TransformedDistribution(base, bijector_for(interval(2.0, 5.0)))
        assert td.support == unit_interval

    def test_round_trip_drifts_for_chain_bijectors(self):
        # ``bijector_for(greater_than(...))`` and
        # ``bijector_for(positive_definite)`` both return ``tfb.Chain``
        # whose outermost bijectors (Shift, CholeskyOuterProduct) aren't
        # in the forward map; ``support`` falls through to ``real``.
        base = Normal(loc=0.0, scale=1.0, name="x")
        td_gt = TransformedDistribution(base, bijector_for(greater_than(3.0)))
        assert td_gt.support == real


# ---------------------------------------------------------------------------
# Custom registration & override
# ---------------------------------------------------------------------------


class TestCustomization:
    def test_register_custom_constraint(self):
        class _MyPositive(Constraint):
            def check(self, value):
                return jnp.asarray(value) > 0

        register_bijector(_MyPositive, lambda c: tfb.Softplus())
        try:
            bij = bijector_for(_MyPositive())
            assert isinstance(bij, tfb.Softplus)
        finally:
            unregister_bijector(_MyPositive)

    def test_instance_override(self):
        """Registering on a singleton overrides the type-level default,
        and removing the override restores it."""
        register_bijector(positive, lambda c: tfb.Softplus())
        try:
            assert isinstance(bijector_for(positive), tfb.Softplus)
        finally:
            unregister_bijector(positive)
        # Restored: type-level Exp default is back.
        assert isinstance(bijector_for(positive), tfb.Exp)

    def test_unregister_is_noop_for_missing_key(self):
        class _NotRegistered(Constraint):
            def check(self, value):
                return jnp.asarray(value) >= 0

        # Should not raise.
        unregister_bijector(_NotRegistered)
