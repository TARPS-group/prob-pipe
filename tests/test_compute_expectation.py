"""Tests for the @compute_expectation decorator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.core.protocols import compute_expectation, SupportsExpectation


class _MockDist(SupportsExpectation):
    """Minimal distribution that supports expectation via MC."""

    _sampling_cost = "low"
    _preferred_orchestration = None

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=False):
        # Deterministic: evaluate at known "samples"
        samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        values = jax.vmap(f)(samples)
        return jnp.mean(values)

    @compute_expectation
    def _mean(self):
        return lambda x: x

    @compute_expectation
    def _variance_like(self):
        return lambda x: x ** 2


class TestComputeExpectation:
    def test_mean_via_compute_expectation(self):
        dist = _MockDist()
        result = dist._mean()
        np.testing.assert_allclose(result, 3.0)

    def test_custom_function(self):
        dist = _MockDist()
        result = dist._variance_like()
        # E[x^2] for x in {1,2,3,4,5} = (1+4+9+16+25)/5 = 11
        np.testing.assert_allclose(result, 11.0)
