"""Contracts of the mixture family (VII.3), pending its implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal, Normal, OutputSpec
from probpipe.distributions._capabilities import SupportsLogProb, SupportsSampling
from probpipe.families import MixtureDistribution


def _components():
    return [
        MultivariateNormal("a", jnp.zeros(2), cov=jnp.eye(2)),
        MultivariateNormal("b", jnp.array([2.0, -1.0]), cov=2.0 * jnp.eye(2)),
    ]


@pytest.mark.pending(reason="the mixture shares its components' event declaration")
def test_the_components_share_one_event_declaration_and_labels_may_differ():
    first = Normal("a", 0.0, 1.0, event_spec=OutputSpec(x=None))
    second = Normal("b", 1.0, 1.0, event_spec=OutputSpec(x=None))
    mixture = MixtureDistribution("m", [first, second], jnp.array([0.5, 0.5]))
    assert mixture.event_spec == first.event_spec


@pytest.mark.pending(reason="components whose declarations differ are refused")
def test_components_with_different_declarations_raise():
    with pytest.raises(ValueError):
        MixtureDistribution(
            "m", [Normal("a", 0.0, 1.0), Normal("b", 0.0, 1.0)], jnp.array([0.5, 0.5])
        )


@pytest.mark.pending(reason="the mixture's weights are nonnegative and sum to one")
def test_weights_that_do_not_sum_to_one_raise():
    with pytest.raises(ValueError):
        MixtureDistribution("m", _components(), jnp.array([0.5, 0.7]))


@pytest.mark.pending(reason="the mixture samples and scores when every component does")
def test_the_log_density_is_the_weighted_log_sum_exp():
    components = _components()
    weights = jnp.array([0.3, 0.7])
    mixture = MixtureDistribution("m", components, weights)
    assert isinstance(mixture, SupportsSampling)
    assert isinstance(mixture, SupportsLogProb)
    x = jnp.array([0.5, 0.5])
    expected = jax.scipy.special.logsumexp(
        jnp.log(weights) + jnp.stack([c._log_prob(x) for c in components])
    )
    np.testing.assert_allclose(mixture._log_prob(x), expected, rtol=1e-6)


@pytest.mark.pending(reason="the mixture's moments combine componentwise")
def test_the_moments_combine_componentwise():
    components = _components()
    weights = jnp.array([0.3, 0.7])
    mixture = MixtureDistribution("m", components, weights)
    means = jnp.stack([c._mean() for c in components])
    mean = weights @ means
    second = sum(
        w * (jnp.asarray(c._cov()) + jnp.outer(m, m))
        for w, c, m in zip(weights, components, means, strict=True)
    )
    np.testing.assert_allclose(mixture._mean(), mean, rtol=1e-6)
    np.testing.assert_allclose(
        jnp.asarray(mixture._cov().to_dense()), second - jnp.outer(mean, mean), rtol=1e-6
    )
