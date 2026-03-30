"""Tests for TransportMap and pushforward dispatch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    Normal,
    LogNormal,
    EmpiricalDistribution,
    BroadcastDistribution,
    TransportMap,
    TFPBijector,
    BijectorTransformedDistribution,
    PushforwardRule,
    PushforwardMethod,
    PushforwardInfo,
    pushforward_registry,
    pushforward,
)


# ---------------------------------------------------------------------------
# Concrete TransportMap for testing
# ---------------------------------------------------------------------------


class SquareMap(TransportMap):
    """f(x) = x ** 2.  Not invertible, so not a bijector."""

    def forward(self, value):
        return value ** 2


class ShiftMap(TransportMap):
    """f(x) = x + c."""

    def __init__(self, c: float):
        self._c = c

    def forward(self, value):
        return value + self._c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# TransportMap.forward and __call__
# ---------------------------------------------------------------------------


class TestForward:
    def test_forward_scalar(self):
        f = SquareMap()
        assert f.forward(3.0) == 9.0

    def test_call_delegates_to_forward(self):
        f = SquareMap()
        assert f(3.0) == 9.0

    def test_forward_array(self):
        f = ShiftMap(10.0)
        x = jnp.array([1.0, 2.0, 3.0])
        result = f(x)
        expected = jnp.array([11.0, 12.0, 13.0])
        assert jnp.allclose(result, expected)

    def test_repr(self):
        f = SquareMap()
        assert "SquareMap" in repr(f)


# ---------------------------------------------------------------------------
# Pushforward with sampling fallback
# ---------------------------------------------------------------------------


class TestSamplingFallback:
    def test_pushforward_returns_empirical(self, key):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        result = f.pushforward(dist, key=key, num_samples=100)
        assert isinstance(result, EmpiricalDistribution)

    def test_pushforward_samples_are_positive(self, key):
        """x^2 is always non-negative."""
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        result = f.pushforward(dist, key=key, num_samples=200)
        assert jnp.all(result.samples >= 0)

    def test_pushforward_shift_mean(self, key):
        """Mean of N(0,1) shifted by 5 should be ~5."""
        f = ShiftMap(5.0)
        dist = Normal(loc=0.0, scale=1.0)
        result = f.pushforward(dist, key=key, num_samples=5000)
        from probpipe import mean
        assert jnp.isclose(mean(result), 5.0, atol=0.2)

    def test_strategy_sampling_forces_empirical(self, key):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        result = bij.pushforward(
            dist, strategy="sampling", key=key, num_samples=50
        )
        assert isinstance(result, EmpiricalDistribution)

    def test_pushforward_provenance_attached(self, key):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        result = f.pushforward(dist, key=key, num_samples=50)
        assert result.source is not None
        assert result.source.operation == "pushforward"
        assert result.source.parents == (dist,)
        assert "sample" in result.source.metadata["method"]


# ---------------------------------------------------------------------------
# Strategy parameter
# ---------------------------------------------------------------------------


class TestStrategy:
    def test_closed_form_raises_when_no_rule(self, key):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        with pytest.raises(ValueError, match="closed_form"):
            f.pushforward(dist, strategy="closed_form")

    def test_change_of_variables_raises_for_non_bijector(self, key):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        with pytest.raises(ValueError, match="change_of_variables"):
            f.pushforward(dist, strategy="change_of_variables")

    def test_invalid_strategy_raises(self, key):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        with pytest.raises(ValueError, match="Unknown strategy"):
            f.pushforward(dist, strategy="magic")


# ---------------------------------------------------------------------------
# Registry: check()
# ---------------------------------------------------------------------------


class TestRegistryCheck:
    def test_check_bijector_returns_cov_info(self):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        bij = TFPBijector(tfb.Exp())
        dist = Normal(loc=0.0, scale=1.0)
        info = pushforward_registry.check(bij, dist)
        assert info.feasible
        # Exp+Normal matches ExpNormalRule (closed_form) at priority 10,
        # above change-of-variables at priority 0
        assert info.method == PushforwardMethod.CLOSED_FORM

    def test_check_generic_map_returns_sample_info(self):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        info = pushforward_registry.check(f, dist)
        assert info.feasible
        assert info.method == PushforwardMethod.SAMPLE

    def test_check_with_strategy_filter(self):
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        info = pushforward_registry.check(f, dist, strategy="closed_form")
        assert not info.feasible


# ---------------------------------------------------------------------------
# Top-level pushforward() function
# ---------------------------------------------------------------------------


class TestPushforwardOp:
    def test_exp_normal_closed_form(self):
        """pushforward(Exp, Normal) → LogNormal via closed-form rule."""
        import tensorflow_probability.substrates.jax.bijectors as tfb

        result = pushforward(TFPBijector(tfb.Exp()), Normal(0, 1))
        assert isinstance(result, LogNormal)

    def test_plain_callable(self, key):
        """pushforward(lambda, dist) wraps in _CallableTransportMap."""
        result = pushforward(lambda x: x ** 2, Normal(0, 1), key=key)
        assert isinstance(result, EmpiricalDistribution)

    def test_return_joint_returns_broadcast_dist(self, key):
        """return_joint=True → BroadcastDistribution."""
        import tensorflow_probability.substrates.jax.bijectors as tfb

        result = pushforward(
            TFPBijector(tfb.Exp()), Normal(0, 1),
            return_joint=True, key=key,
        )
        assert isinstance(result, BroadcastDistribution)

    def test_return_joint_has_exact_marginal(self, key):
        """Closed-form + return_joint stores exact output marginal."""
        import tensorflow_probability.substrates.jax.bijectors as tfb

        result = pushforward(
            TFPBijector(tfb.Exp()), Normal(0, 1),
            return_joint=True, key=key,
        )
        assert hasattr(result, "_exact_output_marginal")
        assert isinstance(result._exact_output_marginal, LogNormal)

    def test_sampling_return_joint(self, key):
        """Sampling fallback with return_joint."""
        result = pushforward(
            lambda x: x ** 2, Normal(0, 1),
            return_joint=True, key=key,
        )
        assert isinstance(result, BroadcastDistribution)

    def test_default_num_samples(self, key):
        """Default num_samples uses WorkflowFunction.DEFAULT_N_BROADCAST_SAMPLES."""
        result = pushforward(lambda x: x ** 2, Normal(0, 1), key=key)
        assert result.n == 128  # DEFAULT_N_BROADCAST_SAMPLES

    def test_custom_num_samples(self, key):
        """Custom num_samples is respected."""
        result = pushforward(lambda x: x ** 2, Normal(0, 1), key=key, num_samples=50)
        assert result.n == 50


# ---------------------------------------------------------------------------
# Registry: rule decorator
# ---------------------------------------------------------------------------


class TestFunctionalRule:
    def test_register_and_dispatch(self, key):
        """Register a custom closed-form rule and verify it fires."""

        @pushforward_registry.rule(
            ShiftMap,
            Normal,
            method=PushforwardMethod.CLOSED_FORM,
            priority=20,
            description="Shift a Normal by a constant",
        )
        def _(m, d, **kw):
            return Normal(loc=d.loc + m._c, scale=d.scale)

        f = ShiftMap(3.0)
        dist = Normal(loc=1.0, scale=2.0)
        result = f.pushforward(dist)

        # Should return a Normal, not EmpiricalDistribution
        assert isinstance(result, Normal)
        assert jnp.isclose(result.loc, 4.0)
        assert jnp.isclose(result.scale, 2.0)

        # check() should report closed_form
        info = pushforward_registry.check(f, dist)
        assert info.method == PushforwardMethod.CLOSED_FORM

    def test_rule_provenance(self, key):
        """Provenance auto-attached when rule doesn't set it."""
        f = ShiftMap(1.0)
        dist = Normal(loc=0.0, scale=1.0)
        result = f.pushforward(dist)
        assert result.source is not None
        assert result.source.operation == "pushforward"


# ---------------------------------------------------------------------------
# Registry: custom PushforwardRule subclass
# ---------------------------------------------------------------------------


class TestCustomRule:
    def test_custom_rule_subclass(self, key):
        class SquareNormalRule(PushforwardRule):
            def map_types(self):
                return (SquareMap,)

            def dist_types(self):
                return (Normal,)

            def check(self, transport_map, dist):
                return PushforwardInfo(
                    feasible=True,
                    method=PushforwardMethod.CLOSED_FORM,
                    description="Custom square-normal rule",
                )

            def apply(self, transport_map, dist, **kwargs):
                return EmpiricalDistribution(
                    jnp.array([1.0, 2.0, 3.0])
                )

            @property
            def priority(self):
                return 50  # high priority

        pushforward_registry.register(SquareNormalRule())
        f = SquareMap()
        dist = Normal(loc=0.0, scale=1.0)
        result = f.pushforward(dist)
        # Should dispatch to our custom rule
        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 3
