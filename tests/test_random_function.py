"""Tests for RandomFunction and ArrayRandomFunction base classes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    Normal,
    RandomFunction,
    ArrayRandomFunction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Minimal concrete subclasses for testing
# ---------------------------------------------------------------------------


class _MinimalRandomFunction(RandomFunction):
    """Minimal concrete RandomFunction for testing."""

    def __call__(self, x):
        return Normal(loc=jnp.zeros(3), scale=jnp.ones(3))


class _MinimalArrayRF(ArrayRandomFunction):
    """Minimal concrete ArrayRandomFunction for testing."""

    def predict(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)
        mean = jnp.zeros((*extra_batch, n))
        return Normal(loc=mean, scale=jnp.ones_like(mean))


class _JointCapableRF(ArrayRandomFunction):
    """ArrayRandomFunction that supports joint_inputs."""

    supports_joint_inputs = True

    def predict(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)
        return Normal(
            loc=jnp.zeros((*extra_batch, n)),
            scale=jnp.ones((*extra_batch, n)),
        )


# ---------------------------------------------------------------------------
# RandomFunction tests
# ---------------------------------------------------------------------------


class TestRandomFunction:
    """Tests for the RandomFunction base class."""

    def test_cannot_instantiate_directly(self):
        """RandomFunction is abstract — __call__ must be implemented."""
        with pytest.raises(TypeError, match="abstract"):
            RandomFunction()

    def test_minimal_subclass_instantiates(self):
        rf = _MinimalRandomFunction()
        assert isinstance(rf, RandomFunction)
        assert isinstance(rf, Distribution)

    def test_call_returns_distribution(self):
        rf = _MinimalRandomFunction()
        dist = rf(jnp.array([1.0, 2.0, 3.0]))
        assert isinstance(dist, Distribution)

    def test_sample_raises(self, key):
        rf = _MinimalRandomFunction()
        with pytest.raises(NotImplementedError, match="does not support sampling"):
            rf.sample(key)

    def test_sample_with_shape_raises(self, key):
        rf = _MinimalRandomFunction()
        with pytest.raises(NotImplementedError, match="does not support sampling"):
            rf.sample(key, sample_shape=(5,))

    def test_log_prob_raises(self):
        rf = _MinimalRandomFunction()
        with pytest.raises(NotImplementedError):
            rf.log_prob(lambda x: x)

    def test_input_shape_raises(self):
        rf = _MinimalRandomFunction()
        with pytest.raises(NotImplementedError, match="input_shape"):
            _ = rf.input_shape

    def test_output_shape_raises(self):
        rf = _MinimalRandomFunction()
        with pytest.raises(NotImplementedError, match="output_shape"):
            _ = rf.output_shape


# ---------------------------------------------------------------------------
# ArrayRandomFunction tests
# ---------------------------------------------------------------------------


class TestArrayRandomFunction:
    """Tests for the ArrayRandomFunction base class."""

    def test_cannot_instantiate_directly(self):
        """ArrayRandomFunction is abstract — predict must be implemented."""
        with pytest.raises(TypeError, match="abstract"):
            ArrayRandomFunction(input_shape=(3,))

    def test_shapes(self):
        rf = _MinimalArrayRF(input_shape=(3,), output_shape=(2,))
        assert rf.input_shape == (3,)
        assert rf.output_shape == (2,)

    def test_default_output_shape(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        assert rf.output_shape == ()

    def test_isinstance_hierarchy(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        assert isinstance(rf, ArrayRandomFunction)
        assert isinstance(rf, RandomFunction)
        assert isinstance(rf, Distribution)

    def test_repr(self):
        rf = _MinimalArrayRF(input_shape=(3,), output_shape=(2,))
        assert "_MinimalArrayRF(input_shape=(3,), output_shape=(2,))" in repr(rf)

    # -- Input validation ---------------------------------------------------

    def test_validate_X_too_few_dims(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            rf(jnp.ones(3))  # needs (n, 3)

    def test_validate_X_wrong_trailing(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        with pytest.raises(ValueError, match="Trailing dimensions"):
            rf(jnp.ones((5, 4)))  # trailing 4 != 3

    def test_validate_X_correct(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        dist = rf(jnp.ones((5, 3)))
        assert isinstance(dist, Distribution)

    def test_validate_X_scalar_input(self):
        """Scalar input_shape=() requires at least 1-D (n,)."""
        rf = _MinimalArrayRF(input_shape=())
        dist = rf(jnp.ones(5))
        assert isinstance(dist, Distribution)

    def test_validate_X_extra_batch(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        dist = rf(jnp.ones((4, 5, 3)))  # extra_batch=(4,)
        assert isinstance(dist, Distribution)

    # -- Joint flag validation ----------------------------------------------

    def test_joint_inputs_rejected_when_unsupported(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        assert not rf.supports_joint_inputs
        with pytest.raises(ValueError, match="joint_inputs"):
            rf(jnp.ones((5, 3)), joint_inputs=True)

    def test_joint_outputs_rejected_when_unsupported(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        assert not rf.supports_joint_outputs
        with pytest.raises(ValueError, match="joint_outputs"):
            rf(jnp.ones((5, 3)), joint_outputs=True)

    def test_joint_inputs_accepted_when_supported(self):
        rf = _JointCapableRF(input_shape=(3,))
        dist = rf(jnp.ones((5, 3)), joint_inputs=True)
        assert isinstance(dist, Distribution)

    # -- _parse_X -----------------------------------------------------------

    def test_parse_X_basic(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        X = jnp.ones((10, 3))
        extra_batch, n = rf._parse_X(X)
        assert extra_batch == ()
        assert n == 10

    def test_parse_X_extra_batch(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        X = jnp.ones((4, 10, 3))
        extra_batch, n = rf._parse_X(X)
        assert extra_batch == (4,)
        assert n == 10

    def test_parse_X_multi_extra_batch(self):
        rf = _MinimalArrayRF(input_shape=(3,))
        X = jnp.ones((2, 4, 10, 3))
        extra_batch, n = rf._parse_X(X)
        assert extra_batch == (2, 4)
        assert n == 10

