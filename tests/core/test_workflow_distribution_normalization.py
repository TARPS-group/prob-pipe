"""Characterization tests for WorkflowFunction distribution normalization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    DistributionArray,
    EmpiricalDistribution,
    KDEDistribution,
    Normal,
    NumericRecordArray,
    log_prob,
    mean,
)
from probpipe.core.node import WorkflowFunction
from probpipe.core.protocols import SupportsLogProb


@pytest.fixture
def normal_external():
    return tfd.Normal(loc=2.0, scale=0.5)


@pytest.fixture
def empirical_dist():
    return EmpiricalDistribution(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        name="x",
    )


@pytest.fixture
def mean_recorder():
    seen = []

    def mean_of_normal(dist: Normal):
        seen.append(dist)
        return mean(dist)

    return mean_of_normal, seen


class TestHintedDistributionConversion:
    def test_concrete_distribution_hint_converts_external_distribution(
        self,
        mean_recorder,
        normal_external,
    ):
        mean_of_normal, seen = mean_recorder
        wf = WorkflowFunction(func=mean_of_normal, dispatch="sequential")

        result = wf(dist=normal_external)

        assert isinstance(seen[0], Normal)
        assert float(result) == 2.0

    def test_protocol_hint_converts_distribution_that_lacks_protocol(self, empirical_dist):
        seen = []

        def log_prob_at_zero(dist: SupportsLogProb):
            seen.append(dist)
            return log_prob(dist, jnp.asarray([0.0]))

        wf = WorkflowFunction(func=log_prob_at_zero, dispatch="sequential")

        result = wf(dist=empirical_dist)

        assert seen[0] is not empirical_dist
        assert isinstance(seen[0], KDEDistribution)
        assert isinstance(seen[0], SupportsLogProb)
        assert jnp.isfinite(jnp.asarray(result)).all()


class TestDistributionArrayHandling:
    def test_zero_dimensional_distribution_array_unwraps_to_scalar_component(
        self,
        mean_recorder,
    ):
        mean_of_normal, seen = mean_recorder
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(),
            loc=jnp.asarray(3.0),
            scale=jnp.asarray(1.0),
            name="zero_d",
        )
        wf = WorkflowFunction(func=mean_of_normal, dispatch="sequential")

        result = wf(dist=da)

        assert isinstance(seen[0], Normal)
        assert float(result) == 3.0

    def test_size_one_distribution_array_remains_a_sweep(self, mean_recorder):
        mean_of_normal, _ = mean_recorder
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(1,),
            loc=jnp.asarray([3.0]),
            scale=jnp.asarray([1.0]),
            name="one_cell",
        )
        wf = WorkflowFunction(func=mean_of_normal, dispatch="sequential")

        result = wf(dist=da)

        assert isinstance(result, NumericRecordArray)
        assert result.batch_shape == (1,)
        np.testing.assert_allclose(result[result.fields[0]], jnp.asarray([3.0]))


class TestUnhintedExternalDistribution:
    def test_unhinted_external_distribution_converts_then_broadcasts(self):
        seen_values = []

        def double(x):
            value = jnp.asarray(x)
            seen_values.append(value)
            return value * 2.0

        wf = WorkflowFunction(
            func=double,
            n_broadcast_samples=8,
            dispatch="sequential",
            seed=42,
        )
        external = tfd.Normal(loc=1.0, scale=0.1)

        result = wf(external)

        assert result.n == 8
        assert [value.shape for value in seen_values] == [()] * 8
        # ``atol=0.0`` is deliberate: multiplication by 2.0 is bit-exact
        # under IEEE 754, so the broadcast output should match the
        # recorded per-call inputs exactly.
        np.testing.assert_allclose(result.samples, 2.0 * jnp.stack(seen_values), atol=0.0)
