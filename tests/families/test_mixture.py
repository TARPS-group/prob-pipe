"""Contracts of the mixture family (VII.3).

A ``MixtureDistribution`` is a convex combination of component laws over one
shared event declaration. It samples when every component samples, it has the
weighted log-sum-exp of the components' densities when every component has a
density, and its moments combine componentwise when every component has them.
"""

from __future__ import annotations

import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    MultivariateNormal,
    Normal,
    NumericRecordBatch,
    NumericRecordSpec,
    OutputSpec,
    workflow_run,
)
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from probpipe.families import MixtureDistribution
from probpipe.operations._moments import mean
from probpipe.operations._sample import sample
from tests.operations._laws import Sampler


def _components():
    """Two bivariate normal laws over one declaration, the component ``x``."""
    return [
        MultivariateNormal("x", jnp.zeros(2), cov=jnp.eye(2), label="a"),
        MultivariateNormal("x", jnp.array([2.0, -1.0]), cov=2.0 * jnp.eye(2), label="b"),
    ]


def test_the_components_share_one_event_declaration_and_labels_may_differ():
    first = Normal("x", 0.0, 1.0, label="a")
    second = Normal("x", 1.0, 1.0, label="b")
    mixture = MixtureDistribution([first, second], jnp.array([0.5, 0.5]), label="m")
    assert mixture.event_spec == first.event_spec
    assert mixture.label == "m"


def test_components_with_different_declarations_raise():
    with pytest.raises(ValueError, match=r"must draw the same event.*Rename them with"):
        MixtureDistribution(
            [Normal("a", 0.0, 1.0), Normal("b", 0.0, 1.0)], jnp.array([0.5, 0.5]), label="m"
        )


def test_components_with_different_shapes_raise_without_a_rename_hint():
    vector = MultivariateNormal("a", jnp.zeros(2), cov=jnp.eye(2))
    with pytest.raises(ValueError, match=r"'a' is an array of shape \(\) in component 0") as info:
        MixtureDistribution([Normal("a", 0.0, 1.0), vector], jnp.array([0.5, 0.5]), label="m")
    assert "with_path_names" not in str(info.value)


def test_a_mixture_has_at_least_one_component():
    with pytest.raises(ValueError, match="at least one component"):
        MixtureDistribution([], jnp.array([]), label="m")


def test_a_component_is_a_law():
    with pytest.raises(TypeError, match="component 1"):
        MixtureDistribution([_components()[0], 3.0], jnp.array([0.5, 0.5]), label="m")


def test_weights_that_do_not_sum_to_one_raise():
    with pytest.raises(ValueError, match=r"must sum to 1, got .* \(sum 1\.2\)"):
        MixtureDistribution(_components(), jnp.array([0.5, 0.7]), label="m")


def test_negative_weights_raise():
    with pytest.raises(ValueError, match="nonnegative"):
        MixtureDistribution(_components(), jnp.array([1.5, -0.5]), label="m")


def test_one_weight_per_component():
    with pytest.raises(ValueError, match=r"has 2 components but got weights of shape \(1,\)"):
        MixtureDistribution(_components(), jnp.array([1.0]), label="m")


def test_the_log_density_is_the_weighted_log_sum_exp():
    components = _components()
    weights = jnp.array([0.3, 0.7])
    mixture = MixtureDistribution(components, weights, label="m")
    assert isinstance(mixture, SupportsSampling)
    assert isinstance(mixture, SupportsLogProb)
    x = jnp.array([0.5, 0.5])
    expected = jax.scipy.special.logsumexp(
        jnp.log(weights) + jnp.stack([c._log_prob(x) for c in components])
    )
    np.testing.assert_allclose(mixture._log_prob(x), expected, rtol=1e-6)


def test_the_log_density_keeps_the_leading_axes_of_a_batch_of_values():
    mixture = MixtureDistribution(_components(), jnp.array([0.3, 0.7]), label="m")
    assert mixture._log_prob(jnp.zeros((4, 2))).shape == (4,)


def test_the_moments_combine_componentwise():
    components = _components()
    weights = jnp.array([0.3, 0.7])
    mixture = MixtureDistribution(components, weights, label="m")
    means = jnp.stack([c._mean() for c in components])
    mean_ = weights @ means
    second = sum(
        w * (c._cov().to_dense() + jnp.outer(m, m))
        for w, c, m in zip(weights, components, means, strict=True)
    )
    np.testing.assert_allclose(mixture._mean(), mean_, rtol=1e-6)
    np.testing.assert_allclose(
        mixture._cov().to_dense(), second - jnp.outer(mean_, mean_), rtol=1e-6
    )


def test_the_variance_is_the_diagonal_of_the_covariance():
    mixture = MixtureDistribution(_components(), jnp.array([0.3, 0.7]), label="m")
    np.testing.assert_allclose(mixture._variance(), jnp.diag(mixture._cov().to_dense()), rtol=1e-6)


def test_the_draws_follow_the_mixture():
    mixture = MixtureDistribution(_components(), jnp.array([0.25, 0.75]), label="m")
    draws = mixture._sample(jax.random.PRNGKey(0), (4000,))
    assert draws.shape == (4000, 2)
    np.testing.assert_allclose(jnp.mean(draws, axis=0), mixture._mean(), atol=0.1)
    one = mixture._sample(jax.random.PRNGKey(1))
    assert one.shape == (2,)


def _counting_components() -> list[Sampler]:
    """Three scalar samplers at locations 0, 10, and 20, which record each sample shape."""
    return [Sampler("x", 0.0), Sampler("x", 10.0), Sampler("x", 20.0)]


def _draw_count(component: Sampler) -> int:
    """How many draws *component* has been asked for."""
    return sum(math.prod(shape) for shape in component.shapes)


def test_each_component_draws_as_many_times_as_it_is_chosen():
    components = _counting_components()
    mixture = MixtureDistribution(components, jnp.array([0.5, 0.5, 0.0]), label="m")
    draws = mixture._sample(jax.random.PRNGKey(0), (200,))
    assert draws.shape == (200,)
    counts = [_draw_count(component) for component in components]
    assert sum(counts) == 200
    assert counts[2] == 0


def test_each_draw_comes_from_its_chosen_component():
    """The chosen components' locations, 10 apart, show which component drew each value."""
    mixture = MixtureDistribution(_counting_components(), jnp.array([0.2, 0.3, 0.5]), label="m")
    draws = np.asarray(mixture._sample(jax.random.PRNGKey(3), (20, 100)))
    assert draws.shape == (20, 100)
    nearest = np.rint(draws / 10.0)
    assert set(np.unique(nearest)) <= {0.0, 1.0, 2.0}
    shares = [np.mean(nearest == index) for index in range(3)]
    np.testing.assert_allclose(shares, [0.2, 0.3, 0.5], atol=4 * np.sqrt(0.25 / 2000))


def test_one_draw_draws_one_component_once():
    components = _counting_components()
    one = MixtureDistribution(components, jnp.array([0.2, 0.3, 0.5]), label="m")._sample(
        jax.random.PRNGKey(4)
    )
    assert np.shape(one) == ()
    assert sum(_draw_count(component) for component in components) == 1


def test_a_traced_draw_still_follows_the_mixture():
    mixture = MixtureDistribution(_components(), jnp.array([0.25, 0.75]), label="m")
    draws = jax.jit(lambda key: mixture._sample(key, (4000,)))(jax.random.PRNGKey(0))
    np.testing.assert_allclose(jnp.mean(draws, axis=0), mixture._mean(), atol=0.1)


def test_it_claims_what_every_component_claims():
    """An empirical component has moments but no density, so the mixture has none."""
    empirical = EmpiricalDistribution(
        jnp.array([[0.0, 0.0], [1.0, 1.0]]), event_spec=OutputSpec(x=None), label="e"
    )
    mixture = MixtureDistribution([_components()[0], empirical], jnp.array([0.5, 0.5]), label="m")
    assert isinstance(mixture, SupportsSampling)
    assert isinstance(mixture, SupportsMean)
    assert isinstance(mixture, SupportsVariance)
    assert isinstance(mixture, SupportsCovariance)
    assert not isinstance(mixture, SupportsLogProb)
    np.testing.assert_allclose(mixture._mean(), jnp.array([0.25, 0.25]), rtol=1e-6)


def test_a_record_event_combines_leaf_by_leaf():
    spec = NumericRecordSpec(a=(), b=())
    first = EmpiricalDistribution(
        NumericRecordBatch(
            {"a": jnp.zeros(3), "b": jnp.ones(3)},
            "atom",
            element_spec=spec,
            label="p",
        ),
        label="p",
    )
    second = EmpiricalDistribution(
        NumericRecordBatch(
            {"a": jnp.ones(3), "b": jnp.ones(3)},
            "atom",
            element_spec=spec,
            label="q",
        ),
        label="q",
    )
    mixture = MixtureDistribution([first, second], jnp.array([0.5, 0.5]), label="m")
    moment = mixture._mean()
    np.testing.assert_allclose(moment["a"], 0.5)
    np.testing.assert_allclose(moment["b"], 1.0)
    draws = mixture._sample(jax.random.PRNGKey(2), (5,))
    assert set(draws) == {"a", "b"} and draws["a"].shape == (5,)


def test_the_sample_operation_draws_the_mixture():
    mixture = MixtureDistribution(_components(), jnp.array([0.5, 0.5]), label="m")
    with workflow_run(seed=0):
        draws = sample(mixture, sample_shape=(3,))
    assert draws.batch_shape == (3,)


def test_a_mixture_pickles():
    mixture = MixtureDistribution(_components(), jnp.array([0.3, 0.7]), label="m")
    restored = pickle.loads(pickle.dumps(mixture))
    assert isinstance(restored, SupportsLogProb)
    np.testing.assert_allclose(restored._mean(), mixture._mean(), rtol=1e-6)


def test_the_monte_carlo_mean_of_a_law_over_laws_is_the_mixture_of_its_draws():
    """A law whose draws are laws has as its mean the equally weighted mixture of its draws."""
    atoms = np.empty(3, dtype=object)
    for index, location in enumerate((0.0, 1.0, 2.0)):
        atoms[index] = Normal("x", location, 1.0)
    from probpipe import DistributionBatch

    laws = EmpiricalDistribution(
        DistributionBatch(
            atoms,
            "law",
            label="laws",
        ),
        component="laws",
    )
    with workflow_run(seed=0):
        estimate = mean.with_options(n_broadcast_samples=60)(laws)
    assert isinstance(estimate, SupportsMean)
    assert estimate.event_spec == atoms[0].event_spec
    assert 0.0 <= float(jnp.asarray(estimate._mean())) <= 2.0
