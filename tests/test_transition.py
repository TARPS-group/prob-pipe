"""Tests for probpipe.core.transition — iterate, with_conversion, with_resampling."""

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    Distribution,
    EmpiricalDistribution,
    MultivariateNormal,
    Provenance,
    iterate,
    mean,
    with_conversion,
    with_resampling,
    IncrementalConditioner,
)
from probpipe.core.node import WorkflowFunction


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def initial():
    """A simple 2-D EmpiricalDistribution centered at zero."""
    return EmpiricalDistribution(jnp.zeros((50, 2)), name="initial")


def shift_step(dist, offset):
    """Shift all samples by a scalar. Returns a bare Distribution."""
    # ``dist.samples`` is a single-field NumericRecord; pull the field
    # for raw-array arithmetic.
    field = dist.samples.fields[0]
    samples = dist.samples[field] + offset
    return EmpiricalDistribution(samples, name=field)


def provenance_step(dist, value):
    """A step that sets its own provenance."""
    field = dist.samples.fields[0]
    samples = dist.samples[field] + value
    new_dist = EmpiricalDistribution(samples, name=field)
    new_dist.with_source(
        Provenance("custom_step", parents=(dist,), metadata={"value": value})
    )
    return new_dist


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------


class TestIterate:
    def test_basic(self, initial):
        """iterate returns a DistributionArray including the initial.

        Post issue #130 PR 1.5, WorkflowFunction outputs whose function
        body returns a Python list of Distributions get wrapped as a
        ``DistributionArray`` (the stacked-collection counterpart to
        ``list[Distribution]``), so indexing, iteration, and len all
        still work.
        """
        from probpipe import DistributionArray
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert isinstance(dists, DistributionArray)
        assert len(dists) == 3  # initial + 2 steps
        assert dists[0] is initial
        assert all(isinstance(d, Distribution) for d in dists)

    def test_values(self, initial):
        """Step results have correct sample values."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert jnp.allclose(dists[1].samples[dists[1].samples.fields[0]], jnp.ones((50, 2)))
        assert jnp.allclose(dists[2].samples[dists[2].samples.fields[0]], jnp.full((50, 2), 3.0))

    def test_provenance_auto_attach(self, initial):
        """Provenance is auto-attached when step function doesn't set it."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0])
        dist = dists[1]
        assert dist.source is not None
        assert dist.source.operation == "iterate"
        assert dist.source.metadata["step"] == 0
        assert dist.source.parents == (initial,)

    def test_provenance_preserved(self, initial):
        """Provenance set by step function is not overwritten."""
        dists = iterate(step_fn=provenance_step, initial=initial, inputs=[1.0])
        dist = dists[1]
        assert dist.source.operation == "custom_step"
        assert dist.source.metadata["value"] == 1.0

    def test_provenance_chain(self, initial):
        """Each step's provenance points to the previous distribution."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0, 3.0])
        assert dists[1].source.parents == (initial,)
        assert dists[2].source.parents == (dists[1],)
        assert dists[3].source.parents == (dists[2],)

    def test_callback(self, initial):
        """Callback receives correct (index, dist) pairs."""
        recorded = []

        def cb(i, dist):
            recorded.append((i, float(dist.samples[dist.samples.fields[0]][0, 0])))

        iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0, 3.0], callback=cb)
        assert len(recorded) == 3
        assert recorded[0] == (0, 1.0)
        assert recorded[1] == (1, 3.0)
        assert recorded[2] == (2, 6.0)

    def test_callback_early_stop(self, initial):
        """Callback returning False truncates iteration."""
        def stop_after_one(i, dist):
            if i >= 1:
                return False

        dists = iterate(
            step_fn=shift_step, initial=initial, inputs=[1.0, 2.0, 3.0, 4.0],
            callback=stop_after_one,
        )
        # initial + steps 0 and 1 (stops after callback for step 1)
        assert len(dists) == 3

    def test_empty_inputs(self, initial):
        """Empty inputs returns list with only the initial distribution."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[])
        assert len(dists) == 1
        assert dists[0] is initial

    def test_bad_return_type(self, initial):
        """Non-Distribution return raises TypeError."""
        def bad_step(dist, inp):
            return "not a distribution"

        with pytest.raises(TypeError, match="returned str"):
            iterate(step_fn=bad_step, initial=initial, inputs=[1])

    def test_final_is_last(self, initial):
        """dists[-1] is the final distribution."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert jnp.allclose(dists[-1].samples[dists[-1].samples.fields[0]], jnp.full((50, 2), 3.0))


# ---------------------------------------------------------------------------
# with_conversion
# ---------------------------------------------------------------------------


class TestWithConversion:
    def test_returns_workflow_function(self):
        """with_conversion returns a WorkflowFunction."""
        step = with_conversion(shift_step, MultivariateNormal)
        assert isinstance(step, WorkflowFunction)
        assert "with_conversion" in step._name
        assert "shift_step" in step._name
        assert "MultivariateNormal" in step._name

    def test_converts_output(self, initial):
        """Output is converted to target type."""
        step = with_conversion(shift_step, MultivariateNormal)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert isinstance(dists[-1], MultivariateNormal)

    def test_pre_conversion_in_provenance_parents(self, initial):
        """Pre-conversion distribution is accessible via provenance parents."""
        step = with_conversion(shift_step, MultivariateNormal)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        converted = dists[-1]
        # The converter sets provenance with the source dist as parent
        assert converted.source is not None
        assert len(converted.source.parents) > 0

    def test_multi_step_stays_parametric(self):
        """Each step produces a parametric distribution usable as next prior."""
        from probpipe import sample as pp_sample

        def parametric_step(dist, shift):
            key = jax.random.PRNGKey(42)
            samples = jnp.asarray(pp_sample(dist, key=key, sample_shape=(50,))) + shift
            return EmpiricalDistribution(samples, name="x")

        initial = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        step = with_conversion(parametric_step, MultivariateNormal)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0, 2.0, 3.0])
        for d in dists[1:]:
            assert isinstance(d, MultivariateNormal)


# ---------------------------------------------------------------------------
# with_resampling
# ---------------------------------------------------------------------------


class TestWithResampling:
    def test_returns_workflow_function(self):
        """with_resampling returns a WorkflowFunction."""
        step = with_resampling(shift_step, ess_threshold=0.5)
        assert isinstance(step, WorkflowFunction)
        assert "with_resampling" in step._name
        assert "shift_step" in step._name

    def test_no_resample_uniform(self):
        """Uniform weights -> no resampling (ESS = N)."""
        initial = EmpiricalDistribution(jnp.zeros((100, 2)), name="x")
        step = with_resampling(shift_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        # No resampling occurred, so no "resample" provenance
        assert dists[-1].source.operation != "resample"

    def test_resample_degenerate(self):
        """Highly non-uniform weights -> resampling triggered."""
        n = 100
        log_w = jnp.full(n, -100.0).at[0].set(0.0)
        samples = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(samples, log_weights=log_w, name="x")

        initial = EmpiricalDistribution(jnp.zeros((n, 2)), name="x")
        step = with_resampling(weighted_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[0.0])
        resampled = dists[-1]
        assert resampled.is_uniform
        assert resampled.source.operation == "resample"

    def test_resample_stores_ess_in_metadata(self):
        """Pre-resampling ESS is stored in provenance metadata."""
        n = 50
        log_w = jnp.full(n, -100.0).at[0].set(0.0)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(jnp.zeros((n, 2)), log_weights=log_w, name="x")

        initial = EmpiricalDistribution(jnp.zeros((n, 2)), name="x")
        step = with_resampling(weighted_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[0.0])
        resampled = dists[-1]
        assert "ess" in resampled.source.metadata
        assert "ess_ratio" in resampled.source.metadata
        assert resampled.source.metadata["ess_ratio"] < 0.5

    def test_non_empirical_passthrough(self):
        """Non-EmpiricalDistribution passes through unchanged."""
        initial = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")

        def mvn_step(dist, inp):
            return MultivariateNormal(loc=jnp.ones(2) * inp, cov=jnp.eye(2), name="z")

        step = with_resampling(mvn_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert isinstance(dists[-1], MultivariateNormal)

    def test_deterministic_seed(self):
        """Resampling is deterministic across repeated calls with same seed."""
        n = 100
        log_w = jnp.full(n, -100.0).at[0].set(0.0)
        samples = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(samples, log_weights=log_w, name="x")

        initial = EmpiricalDistribution(jnp.zeros((n, 2)), name="x")

        step1 = with_resampling(weighted_step, ess_threshold=0.5, seed=42)
        dists1 = iterate(step_fn=step1, initial=initial, inputs=[0.0, 0.0])

        step2 = with_resampling(weighted_step, ess_threshold=0.5, seed=42)
        dists2 = iterate(step_fn=step2, initial=initial, inputs=[0.0, 0.0])

        assert jnp.allclose(dists1[1].samples[dists1[1].samples.fields[0]], dists2[1].samples[dists2[1].samples.fields[0]])
        assert jnp.allclose(dists1[2].samples, dists2[2].samples)


# ---------------------------------------------------------------------------
# IncrementalConditioner
# ---------------------------------------------------------------------------


def _mock_condition_fn(model, data, **kwargs):
    """Conditioning function for testing: return EmpiricalDistribution near data mean."""
    data_mean = jnp.mean(jnp.asarray(data), axis=0)
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(50, data_mean.shape[0]))
    samples = data_mean[None, :] + noise * 0.1
    return EmpiricalDistribution(samples, name="x")


class _SimpleLikelihood:
    def log_likelihood(self, params, data):
        return -0.5 * jnp.sum((data - params) ** 2)


class TestIncrementalConditioner:
    def test_update_single_batch(self):
        """update() conditions on a single data batch, updates state."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0, name="prior")
        conditioner = IncrementalConditioner(
            prior, _SimpleLikelihood(), condition_fn=_mock_condition_fn,
        )
        assert conditioner.curr_posterior is prior

        data = jnp.ones((10, 2)) * 2.0
        posterior = conditioner.update(data=data)

        assert isinstance(posterior, Distribution)
        assert conditioner.curr_posterior is posterior

    def test_update_successive(self):
        """Successive update() calls chain posteriors."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0, name="prior")
        conditioner = IncrementalConditioner(
            prior, _SimpleLikelihood(), condition_fn=_mock_condition_fn,
        )
        post1 = conditioner.update(data=jnp.ones((10, 2)))
        post2 = conditioner.update(data=jnp.ones((10, 2)) * 2.0)
        assert conditioner.curr_posterior is post2
        assert post1 is not post2

    def test_update_all(self):
        """update_all() iterates over batches, returns DistributionArray,
        updates state."""
        from probpipe import DistributionArray
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0, name="prior")
        conditioner = IncrementalConditioner(
            prior, _SimpleLikelihood(), condition_fn=_mock_condition_fn,
        )
        batches = [jnp.ones((10, 2)) * i for i in [1.0, 2.0, 3.0]]
        dists = conditioner.update_all(data_batches=batches)

        assert isinstance(dists, DistributionArray)
        assert len(dists) == 4  # prior + 3 steps
        assert dists[0] is prior
        assert conditioner.curr_posterior is dists[-1]

    def test_step_property(self):
        """step property exposes the step function for use with iterate."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0, name="prior")
        conditioner = IncrementalConditioner(
            prior, _SimpleLikelihood(), condition_fn=_mock_condition_fn,
        )
        assert isinstance(conditioner.step, WorkflowFunction)

        # Use .step with iterate
        batches = [jnp.ones((10, 2)) * i for i in [1.0, 2.0]]
        dists = iterate(conditioner.step, prior, batches)
        assert len(dists) == 3
        assert all(isinstance(d, Distribution) for d in dists)


# ---------------------------------------------------------------------------
# Nestability
# ---------------------------------------------------------------------------


class TestNestability:
    def test_nested_iterate(self, initial):
        """A step function can call iterate internally."""
        def inner_step(dist, value):
            field = dist.samples.fields[0]
            return EmpiricalDistribution(
                dist.samples[field] + value, name=field,
            )

        def outer_step(dist, batch):
            """Each outer step runs an inner iterate loop."""
            inner_dists = iterate(inner_step, dist, batch)
            return inner_dists[-1]

        outer_inputs = [[0.1, 0.2], [0.3, 0.4, 0.5]]
        dists = iterate(step_fn=outer_step, initial=initial, inputs=outer_inputs)
        assert len(dists) == 3  # initial + 2 outer steps
        # Total shift: (0.1+0.2) + (0.3+0.4+0.5) = 1.5
        assert jnp.allclose(dists[-1].samples, jnp.full((50, 2), 1.5))
