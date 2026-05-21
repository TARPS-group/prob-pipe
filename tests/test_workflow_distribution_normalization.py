"""Characterization tests for WorkflowFunction distribution normalization."""

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    DistributionArray,
    EmpiricalDistribution,
    Normal,
    NumericRecordArray,
)
from probpipe.core.node import WorkflowFunction
from probpipe.core.protocols import SupportsLogProb


def test_concrete_distribution_hint_converts_external_distribution():
    seen = []

    def mean_of_normal(dist: Normal):
        seen.append(dist)
        return dist._mean()

    wf = WorkflowFunction(func=mean_of_normal, vectorize="loop")
    external = tfd.Normal(loc=2.0, scale=0.5)

    result = wf(dist=external)

    assert isinstance(seen[0], Normal)
    assert float(result) == 2.0


def test_protocol_hint_converts_distribution_that_lacks_protocol():
    seen = []
    empirical = EmpiricalDistribution(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        name="x",
    )

    def log_prob_at_zero(dist: SupportsLogProb):
        seen.append(dist)
        return dist._log_prob(jnp.asarray([0.0]))

    wf = WorkflowFunction(func=log_prob_at_zero, vectorize="loop")

    result = wf(dist=empirical)

    assert seen[0] is not empirical
    assert isinstance(seen[0], SupportsLogProb)
    assert jnp.isfinite(jnp.asarray(result)).all()


def test_zero_dimensional_distribution_array_unwraps_to_scalar_component():
    seen = []

    def mean_of_normal(dist: Normal):
        seen.append(dist)
        return dist._mean()

    da = DistributionArray.from_batched_params(
        Normal,
        batch_shape=(),
        loc=jnp.asarray(3.0),
        scale=jnp.asarray(1.0),
        name="zero_d",
    )
    wf = WorkflowFunction(func=mean_of_normal, vectorize="loop")

    result = wf(dist=da)

    assert isinstance(seen[0], Normal)
    assert float(result) == 3.0


def test_size_one_distribution_array_remains_a_sweep():
    def mean_of_normal(dist: Normal):
        return dist._mean()

    da = DistributionArray.from_batched_params(
        Normal,
        batch_shape=(1,),
        loc=jnp.asarray([3.0]),
        scale=jnp.asarray([1.0]),
        name="one_cell",
    )
    wf = WorkflowFunction(func=mean_of_normal, vectorize="loop")

    result = wf(dist=da)

    assert isinstance(result, NumericRecordArray)
    assert result.batch_shape == (1,)
    np.testing.assert_allclose(result["mean_of_normal"], jnp.asarray([3.0]))


def test_unhinted_external_distribution_converts_then_broadcasts():
    seen_shapes = []

    def double(x):
        seen_shapes.append(jnp.asarray(x).shape)
        return x * 2.0

    wf = WorkflowFunction(
        func=double,
        n_broadcast_samples=8,
        vectorize="loop",
        seed=0,
    )
    external = tfd.Normal(loc=1.0, scale=0.1)

    result = wf(external)

    assert result.n == 8
    assert seen_shapes == [()] * 8
    np.testing.assert_allclose(float(jnp.mean(result.samples)), 2.0, atol=0.25)
