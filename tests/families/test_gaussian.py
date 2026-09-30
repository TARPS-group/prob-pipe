"""Contracts of the Gaussian algebra (VII.6), pending their implementation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal, Normal
from probpipe.distributions import FactoredNumericDistribution
from probpipe.families import (
    FactoredMultivariateGaussian,
    GaussianProcess,
    LinearGaussianConditional,
)
from probpipe.linalg import DenseLinOp


def _rbf(x, y):
    return jnp.exp(-0.5 * (x[:, None, 0] - y[None, :, 0]) ** 2)


def test_the_factored_gaussian_is_a_numeric_factored_joint():
    assert issubclass(FactoredMultivariateGaussian, FactoredNumericDistribution)


@pytest.mark.pending(reason="* derives the factored Gaussian from Gaussian factors")
def test_composition_derives_the_factored_gaussian():
    prior = MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
    observation = LinearGaussianConditional(
        "y", DenseLinOp(jnp.array([[1.0, 0.0], [1.0, 1.0]])), jnp.zeros(2), DenseLinOp(jnp.eye(2))
    )
    assert isinstance(observation * prior, FactoredMultivariateGaussian)


class TestTheGaussianProcess:
    @pytest.mark.pending(reason="a Gaussian process evaluates to its finite-dimensional law")
    def test_evaluation_at_stacked_points_is_a_multivariate_normal(self):
        process = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf)
        X = jnp.array([[0.0], [0.5], [2.0]])
        law = process(X)
        assert isinstance(law, MultivariateNormal)
        np.testing.assert_allclose(law._mean(), jnp.zeros(3), atol=1e-6)
        np.testing.assert_allclose(jnp.asarray(law._cov()), _rbf(X, X), rtol=1e-5)

    @pytest.mark.pending(reason="a Gaussian process evaluates to a Normal at one point")
    def test_evaluation_at_one_point_is_a_normal(self):
        process = GaussianProcess("f", lambda X: jnp.ones(X.shape[0]), _rbf)
        assert isinstance(process(jnp.array([[0.0]])), Normal)

    @pytest.mark.pending(reason="the drawn function's output component defaults to the label")
    def test_the_components_default_to_the_label(self):
        process = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf)
        assert list(process.event_spec.components) == ["f"]
