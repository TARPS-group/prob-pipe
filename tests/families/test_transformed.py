"""Contracts of the evaluation-result families (VII.4), pending their implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Function, MultivariateNormal, Normal, ResolutionError
from probpipe.families import BijectorTransformedDistribution, LinearPushforwardDistribution
from probpipe.linalg import DenseLinOp, LinOp


@pytest.fixture
def base():
    return MultivariateNormal("x", jnp.array([1.0, -1.0]), cov=jnp.array([[2.0, 0.5], [0.5, 1.0]]))


@pytest.fixture
def op():
    return DenseLinOp(jnp.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]]))


class TestTheLinearPushforward:
    @pytest.mark.pending(reason="the pushforward's event is the operator's output type")
    def test_the_event_is_the_operator_output_under_its_own_component(self, base, op):
        pushed = LinearPushforwardDistribution("y", base, op)
        assert list(pushed.event_spec.components) == ["y"]
        assert pushed.event_spec.spec.shape == (3,)

    @pytest.mark.pending(reason="the pushforward's moments delegate exactly")
    def test_the_moments_delegate_through_the_operator(self, base, op):
        pushed = LinearPushforwardDistribution("y", base, op)
        A = op.to_dense()
        np.testing.assert_allclose(pushed._mean(), A @ base._mean(), rtol=1e-6)
        covariance = pushed._cov()
        assert isinstance(covariance, LinOp)
        np.testing.assert_allclose(
            covariance.to_dense(), A @ jnp.asarray(base._cov()) @ A.T, rtol=1e-5
        )

    @pytest.mark.pending(reason="the pushforward samples by pushing base draws through op")
    def test_sampling_pushes_the_base_draws(self, base, op):
        pushed = LinearPushforwardDistribution("y", base, op)
        key = jax.random.PRNGKey(0)
        np.testing.assert_allclose(
            pushed._sample(key), op.to_dense() @ base._sample(key), rtol=1e-6
        )


class TestTheBijectorTransform:
    @staticmethod
    def _exp():
        return Function("exp", jnp.exp)

    @pytest.mark.pending(reason="the transform checks the bijector's claims at construction")
    def test_a_map_without_an_inverse_raises(self):
        with pytest.raises(ResolutionError):
            BijectorTransformedDistribution("y", Normal("x", 0.0, 1.0), self._exp())

    @pytest.mark.pending(reason="the transform's density is the change of variables")
    def test_the_log_density_is_the_change_of_variables(self):
        transformed = BijectorTransformedDistribution("y", Normal("x", 0.0, 1.0), self._exp())
        y = jnp.asarray(2.0)
        expected = Normal("x", 0.0, 1.0)._log_prob(jnp.log(y)) - jnp.log(y)
        np.testing.assert_allclose(transformed._log_prob(y), expected, rtol=1e-6)
