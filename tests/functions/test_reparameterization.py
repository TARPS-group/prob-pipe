"""Contracts of the constraint-to-bijector factory (V.12)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    Function,
    MathematicalDomainError,
    ResolutionError,
    SupportsLogDetJacobian,
    bijector_for,
    boolean,
    greater_than,
    integer_interval,
    interval,
    is_invertible,
    non_negative,
    non_negative_integer,
    positive,
    positive_definite,
    real,
    register_bijector,
    simplex,
    sphere,
    unit_interval,
)
from probpipe.core.constraints import Constraint
from probpipe.functions._reparameterization import _CONSTRAINT_BIJECTOR_REGISTRY, _image


@pytest.fixture
def registry_snapshot():
    """Restore the factory registry, which is module state, after a test registers one."""
    snapshot = dict(_CONSTRAINT_BIJECTOR_REGISTRY)
    try:
        yield
    finally:
        _CONSTRAINT_BIJECTOR_REGISTRY.clear()
        _CONSTRAINT_BIJECTOR_REGISTRY.update(snapshot)


_SUPPORTED = [
    real,
    positive,
    non_negative,
    unit_interval,
    interval(2.0, 5.0),
    greater_than(3.0),
    simplex,
    positive_definite,
]


class TestTheBijectorIsAFunction:
    @pytest.mark.parametrize("constraint", _SUPPORTED, ids=repr)
    def test_it_claims_the_inverse_and_the_log_jacobian(self, constraint):
        bijector = bijector_for(constraint)
        assert isinstance(bijector, Function)
        assert is_invertible(bijector)
        assert isinstance(bijector, SupportsLogDetJacobian)

    @pytest.mark.parametrize("constraint", _SUPPORTED, ids=repr)
    def test_it_records_the_support_it_maps_onto(self, constraint):
        assert _image(bijector_for(constraint)) == constraint

    def test_the_log_jacobian_is_that_of_the_forward_map_at_a_point(self):
        bijector = bijector_for(positive)
        x = jnp.array([-1.0, 0.5, 2.0])
        # exp is elementwise, so log |det J(x)| is the sum of log exp'(x_i) = x_i.
        assert float(bijector._log_det_jacobian(x)) == pytest.approx(float(jnp.sum(x)), rel=1e-6)
        derivative = jax.vmap(jax.grad(bijector.apply))(x)
        np.testing.assert_allclose(jnp.log(derivative), x, rtol=1e-6)


class TestRoundTrip:
    def test_real_identity(self):
        bijector = bijector_for(real)
        x = jnp.array([-3.0, 0.0, 4.5])
        np.testing.assert_allclose(bijector.apply(x), x)
        np.testing.assert_allclose(bijector._inverse(bijector.apply(x)), x)

    def test_positive_exp(self):
        bijector = bijector_for(positive)
        x = jnp.array([-2.0, 0.0, 3.0])
        y = bijector.apply(x)
        np.testing.assert_allclose(y, jnp.exp(x), rtol=1e-6)
        assert jnp.all(positive.check(y))
        np.testing.assert_allclose(bijector._inverse(y), x, atol=1e-5)

    def test_non_negative_softplus(self):
        bijector = bijector_for(non_negative)
        x = jnp.array([-100.0, -1.0, 0.0, 5.0])
        y = bijector.apply(x)
        np.testing.assert_allclose(y, jax.nn.softplus(x), rtol=1e-6)
        assert jnp.all(non_negative.check(y))

    def test_unit_interval_sigmoid(self):
        bijector = bijector_for(unit_interval)
        x = jnp.array([-100.0, 0.0, 100.0])
        y = bijector.apply(x)
        np.testing.assert_allclose(y, jax.nn.sigmoid(x), rtol=1e-6)
        assert jnp.all(unit_interval.check(y))

    def test_interval_sigmoid(self):
        c = interval(2.0, 5.0)
        bijector = bijector_for(c)
        # Moderate scales: float32 saturates the sigmoid past about |x| = 15.
        y = bijector.apply(jnp.array([-3.0, 0.0, 3.0]))
        assert jnp.all(c.check(y))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(y > 2.0) and jnp.all(y < 5.0)
        x0 = jnp.array([0.0, 1.0])
        np.testing.assert_allclose(bijector._inverse(bijector.apply(x0)), x0, atol=1e-5)

    def test_greater_than(self):
        c = greater_than(3.0)
        y = bijector_for(c).apply(jnp.array([-2.0, 0.0, 5.0]))
        assert jnp.all(c.check(y))

    def test_simplex_softmax_centered(self):
        bijector = bijector_for(simplex)
        y = bijector.apply(jnp.array([0.5, -1.0]))  # 2 unconstrained → a point of the 3-simplex
        assert simplex.check(y)
        assert y.shape == (3,)

    def test_the_simplex_log_jacobian_is_taken_in_its_first_coordinates(self):
        """A Dirichlet's density is stated in the simplex's first K - 1 coordinates, so the
        log-Jacobian is that of the map onto them."""
        bijector = bijector_for(simplex)
        z = jnp.array([0.3, -0.2, 0.9])
        jacobian = jax.jacfwd(lambda u: bijector.raw()(u)[:-1])(z)
        expected = jnp.log(jnp.abs(jnp.linalg.det(jacobian)))
        np.testing.assert_allclose(bijector._log_det_jacobian(z), expected, rtol=1e-5)
        backend = bijector._bijector
        x = backend.forward(z)
        np.testing.assert_allclose(
            backend.inverse_log_det_jacobian(x, event_ndims=1), -expected, rtol=1e-5
        )

    def test_positive_definite(self):
        bijector = bijector_for(positive_definite)
        # 6 unconstrained parameters fill a 3x3 lower-triangular L, then L Lᵀ.
        m = bijector.apply(jnp.array([1.0, 0.5, 2.0, -0.3, 0.1, 1.5]))
        assert m.shape == (3, 3)
        assert positive_definite.check(m)
        np.testing.assert_allclose(m, m.T, atol=1e-5)


class TestVectorEventShapes:
    def test_interval_with_vector_bounds(self):
        upper = jnp.array([1.0, 2.0, 3.0])
        # Array bounds do not hash, so the lookup falls through to the type.
        y = bijector_for(interval(jnp.zeros(3), upper)).apply(jnp.array([-2.0, 0.0, 2.0]))
        assert jnp.all(y >= 0.0)
        assert jnp.all(y <= upper)

    def test_positive_pointwise(self):
        x = jnp.array([[-1.0, 2.0], [3.0, -4.0]])
        y = bijector_for(positive).apply(x)
        assert jnp.all(y > 0)
        assert y.shape == x.shape


class TestUnsupported:
    @pytest.mark.parametrize(
        "constraint", [sphere, boolean, non_negative_integer, integer_interval(0, 5)], ids=repr
    )
    def test_a_support_with_no_smooth_bijector_is_a_domain_error(self, constraint):
        with pytest.raises(MathematicalDomainError, match="no smooth bijector"):
            bijector_for(constraint)

    def test_an_unregistered_constraint_is_unresolved(self):
        class MyConstraint(Constraint):
            def check(self, value):
                return jnp.asarray(value) >= 0

        with pytest.raises(ResolutionError, match="No bijector registered"):
            bijector_for(MyConstraint())

    def test_a_factory_whose_map_claims_no_inverse_is_unresolved(self, registry_snapshot):
        class _Unmapped(Constraint):
            def check(self, value):
                return jnp.asarray(value) > 0

        register_bijector(_Unmapped, lambda c: Function("exp", jnp.exp))
        with pytest.raises(ResolutionError, match="SupportsInverse"):
            bijector_for(_Unmapped())


class TestBoundary:
    """The sigmoid and the exponential saturate or overflow at extreme inputs in float32,
    and the factory promises no clipping, so these use the scales of unconstrained
    optimization, within about ten."""

    def test_interval_moderate_inputs(self):
        y = bijector_for(interval(0.0, 1.0)).apply(jnp.array([-5.0, 0.0, 5.0]))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all((y > 0.0) & (y < 1.0))

    def test_positive_moderate_inputs(self):
        y = bijector_for(positive).apply(jnp.array([-5.0, 0.0, 5.0]))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(y > 0)

    def test_greater_than_moderate_inputs(self):
        c = greater_than(-2.5)
        y = bijector_for(c).apply(jnp.array([-5.0, 5.0]))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(c.check(y))


class TestCustomization:
    def test_register_custom_constraint(self, registry_snapshot):
        class _MyPositive(Constraint):
            def check(self, value):
                return jnp.asarray(value) > 0

        register_bijector(_MyPositive, lambda c: tfb.Softplus())
        bijector = bijector_for(_MyPositive())
        np.testing.assert_allclose(bijector.apply(jnp.array([0.0])), jax.nn.softplus(0.0))

    def test_instance_override(self, registry_snapshot):
        """Registering on a singleton overrides the type-level default."""
        register_bijector(positive, lambda c: tfb.Softplus())
        np.testing.assert_allclose(
            bijector_for(positive).apply(jnp.array([0.0])), jax.nn.softplus(0.0)
        )

    def test_a_factory_may_return_a_function(self, registry_snapshot):
        class _Doubled(Function):
            def __init__(self) -> None:
                super().__init__("double", lambda x: 2.0 * x)

            def _inverse(self, y):
                return y / 2.0

            def _log_det_jacobian(self, x):
                return jnp.size(x) * jnp.log(2.0)

        register_bijector(positive, lambda c: _Doubled())
        bijector = bijector_for(positive)
        assert isinstance(bijector, _Doubled)
