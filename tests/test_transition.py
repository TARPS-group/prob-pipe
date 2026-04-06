"""Tests for probpipe.core.transition — StepResult, TransitionTrace, iterate, combinators."""

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    EmpiricalDistribution,
    MultivariateNormal,
    Provenance,
    StepResult,
    TransitionTrace,
    iterate,
    with_approximation,
    with_resampling,
    ConditioningStep,
    IncrementalConditioner,
)
from probpipe.core.node import WorkflowFunction
from probpipe.core.transition import DistributionTransition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def initial():
    """A simple 2-D EmpiricalDistribution centered at zero."""
    return EmpiricalDistribution(jnp.zeros((50, 2)), name="initial")


def trivial_step(dist, shift):
    """Shift all samples by a scalar. Returns a bare Distribution."""
    samples = dist.samples + shift
    return EmpiricalDistribution(samples)


def info_step(dist, data):
    """Returns a StepResult with info containing the data mean."""
    samples = dist.samples + jnp.mean(jnp.asarray(data), axis=0)
    new_dist = EmpiricalDistribution(samples)
    return StepResult(
        distribution=new_dist,
        info={"data_mean": float(jnp.mean(data))},
    )


def provenance_step(dist, value):
    """A step that sets its own provenance."""
    samples = dist.samples + value
    new_dist = EmpiricalDistribution(samples, name="custom")
    new_dist.with_source(
        Provenance("custom_step", parents=(dist,), metadata={"value": value})
    )
    return new_dist


# ---------------------------------------------------------------------------
# Category 1: StepResult / TransitionTrace unit tests
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_construction(self, initial):
        result = StepResult(distribution=initial)
        assert result.distribution is initial
        assert result.info == {}

    def test_construction_with_info(self, initial):
        result = StepResult(distribution=initial, info={"key": "value"})
        assert result.info["key"] == "value"

    def test_frozen(self, initial):
        result = StepResult(distribution=initial)
        with pytest.raises(AttributeError):
            result.distribution = initial


class TestTransitionTrace:
    def test_empty_trace(self, initial):
        trace = TransitionTrace(initial=initial, results=())
        assert len(trace) == 0
        assert trace.final is initial
        assert trace.distributions == [initial]
        assert trace.infos == []

    def test_single_step(self, initial):
        new_dist = EmpiricalDistribution(jnp.ones((50, 2)))
        result = StepResult(distribution=new_dist, info={"a": 1})
        trace = TransitionTrace(initial=initial, results=(result,))
        assert len(trace) == 1
        assert trace.final is new_dist
        assert trace.distributions == [initial, new_dist]
        assert trace.infos == [{"a": 1}]

    def test_multi_step(self, initial):
        results = tuple(
            StepResult(distribution=EmpiricalDistribution(jnp.full((50, 2), float(i))), info={"step": i})
            for i in range(3)
        )
        trace = TransitionTrace(initial=initial, results=results)
        assert len(trace) == 3
        assert trace.final is results[-1].distribution
        assert len(trace.distributions) == 4  # initial + 3 steps

    def test_info_values(self, initial):
        results = tuple(
            StepResult(
                distribution=EmpiricalDistribution(jnp.zeros((10, 2))),
                info={"ess": float(i * 10)},
            )
            for i in range(3)
        )
        trace = TransitionTrace(initial=initial, results=results)
        assert trace.info_values("ess") == [0.0, 10.0, 20.0]

    def test_info_values_missing_key(self, initial):
        results = (
            StepResult(
                distribution=EmpiricalDistribution(jnp.zeros((10, 2))),
                info={},
            ),
        )
        trace = TransitionTrace(initial=initial, results=results)
        with pytest.raises(KeyError):
            trace.info_values("missing")

    def test_getitem(self, initial):
        results = tuple(
            StepResult(
                distribution=EmpiricalDistribution(jnp.full((10, 2), float(i))),
                info={"i": i},
            )
            for i in range(3)
        )
        trace = TransitionTrace(initial=initial, results=results)
        assert trace[0].info["i"] == 0
        assert trace[2].info["i"] == 2
        assert trace[-1].info["i"] == 2

    def test_iter(self, initial):
        """TransitionTrace supports iteration over results."""
        results = tuple(
            StepResult(
                distribution=EmpiricalDistribution(jnp.full((10, 2), float(i))),
                info={"i": i},
            )
            for i in range(3)
        )
        trace = TransitionTrace(initial=initial, results=results)
        collected = list(trace)
        assert len(collected) == 3
        assert all(isinstance(r, StepResult) for r in collected)
        assert collected[0].info["i"] == 0
        assert collected[2].info["i"] == 2

    def test_frozen(self, initial):
        """TransitionTrace is immutable."""
        trace = TransitionTrace(initial=initial, results=())
        with pytest.raises(AttributeError):
            trace.initial = initial


# ---------------------------------------------------------------------------
# Category 2: iterate behavior
# ---------------------------------------------------------------------------


class TestIterate:
    def test_bare_distribution_return(self, initial):
        """Step function returning a bare Distribution is normalized."""
        trace = iterate(step_fn=trivial_step, initial=initial, inputs=[1.0, 2.0])
        assert len(trace) == 2
        # First step shifts by 1.0, second by 2.0 (from the shifted position)
        assert jnp.allclose(trace[0].distribution.samples, jnp.ones((50, 2)))
        assert jnp.allclose(trace[1].distribution.samples, jnp.full((50, 2), 3.0))

    def test_step_result_return(self, initial):
        """Step function returning StepResult preserves info."""
        data_batches = [jnp.ones((10, 2)) * i for i in [1.0, 2.0]]
        trace = iterate(step_fn=info_step, initial=initial, inputs=data_batches)
        assert len(trace) == 2
        assert trace[0].info["data_mean"] == 1.0
        assert trace[1].info["data_mean"] == 2.0

    def test_provenance_auto_attach(self, initial):
        """Provenance is auto-attached when step function doesn't set it."""
        trace = iterate(step_fn=trivial_step, initial=initial, inputs=[1.0])
        dist = trace[0].distribution
        assert dist.source is not None
        assert dist.source.operation == "iterate"
        assert dist.source.metadata["step"] == 0
        assert dist.source.parents == (initial,)

    def test_provenance_preserved(self, initial):
        """Provenance set by step function is not overwritten."""
        trace = iterate(step_fn=provenance_step, initial=initial, inputs=[1.0])
        dist = trace[0].distribution
        assert dist.source.operation == "custom_step"
        assert dist.source.metadata["value"] == 1.0

    def test_provenance_chain(self, initial):
        """Each step's provenance points to the previous step's distribution."""
        trace = iterate(step_fn=trivial_step, initial=initial, inputs=[1.0, 2.0, 3.0])
        # Step 0 parent is initial
        assert trace[0].distribution.source.parents == (initial,)
        # Step 1 parent is step 0's distribution
        assert trace[1].distribution.source.parents == (trace[0].distribution,)
        # Step 2 parent is step 1's distribution
        assert trace[2].distribution.source.parents == (trace[1].distribution,)

    def test_callback(self, initial):
        """Callback receives correct (index, result) pairs."""
        recorded = []

        def cb(i, result):
            recorded.append((i, float(result.distribution.samples[0, 0])))

        iterate(step_fn=trivial_step, initial=initial, inputs=[1.0, 2.0, 3.0], callback=cb)
        assert len(recorded) == 3
        assert recorded[0] == (0, 1.0)
        assert recorded[1] == (1, 3.0)
        assert recorded[2] == (2, 6.0)

    def test_callback_early_stop(self, initial):
        """Callback returning False truncates iteration."""
        def stop_after_one(i, result):
            if i >= 1:
                return False

        trace = iterate(
            step_fn=trivial_step, initial=initial, inputs=[1.0, 2.0, 3.0, 4.0],
            callback=stop_after_one,
        )
        # Should have steps 0 and 1 (stops after callback for step 1 returns False)
        assert len(trace) == 2

    def test_empty_inputs(self, initial):
        """Empty inputs returns trace with only the initial distribution."""
        trace = iterate(step_fn=trivial_step, initial=initial, inputs=[])
        assert len(trace) == 0
        assert trace.final is initial

    def test_bad_return_type(self, initial):
        """Non-Distribution/StepResult return raises TypeError."""
        def bad_step(dist, inp):
            return "not a distribution"

        with pytest.raises(TypeError, match="returned str"):
            iterate(step_fn=bad_step, initial=initial, inputs=[1])

    def test_final_matches_last_step(self, initial):
        """trace.final is the same object as the last step's distribution."""
        trace = iterate(step_fn=trivial_step, initial=initial, inputs=[1.0, 2.0])
        assert trace.final is trace[-1].distribution


# ---------------------------------------------------------------------------
# Category 3: Combinators
# ---------------------------------------------------------------------------


class TestWithApproximation:
    def test_returns_workflow_function(self):
        """with_approximation returns a WorkflowFunction."""
        step = with_approximation(trivial_step, MultivariateNormal)
        assert isinstance(step, WorkflowFunction)
        assert "with_approximation" in step._name
        assert "trivial_step" in step._name
        assert "MultivariateNormal" in step._name

    def test_converts_output(self, initial):
        """Output is converted to target type."""
        step = with_approximation(trivial_step, MultivariateNormal)
        trace = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert isinstance(trace.final, MultivariateNormal)

    def test_preserves_pre_approximation(self, initial):
        """Pre-conversion distribution is stored in info."""
        step = with_approximation(trivial_step, MultivariateNormal)
        trace = iterate(step_fn=step, initial=initial, inputs=[1.0])
        pre = trace[0].info["pre_approximation"]
        assert isinstance(pre, EmpiricalDistribution)

    def test_preserves_inner_info(self, initial):
        """Info from the inner step function is preserved."""
        data = [jnp.ones((10, 2))]
        step = with_approximation(info_step, MultivariateNormal)
        trace = iterate(step_fn=step, initial=initial, inputs=data)
        assert "data_mean" in trace[0].info
        assert "pre_approximation" in trace[0].info

    def test_multi_step_stays_parametric(self):
        """Each step produces a parametric distribution usable as next prior."""
        from probpipe import sample as pp_sample

        def parametric_step(dist, shift):
            """A step that works with any SupportsSampling distribution."""
            key = jax.random.PRNGKey(42)
            samples = pp_sample(dist, key=key, sample_shape=(50,)) + shift
            return EmpiricalDistribution(samples)

        initial = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        step = with_approximation(parametric_step, MultivariateNormal)
        trace = iterate(step_fn=step, initial=initial, inputs=[1.0, 2.0, 3.0])
        for i in range(3):
            assert isinstance(trace[i].distribution, MultivariateNormal)


class TestWithResampling:
    def test_returns_workflow_function(self):
        """with_resampling returns a WorkflowFunction."""
        step = with_resampling(trivial_step, ess_threshold=0.5)
        assert isinstance(step, WorkflowFunction)
        assert "with_resampling" in step._name
        assert "trivial_step" in step._name

    def test_no_resample_uniform(self):
        """Uniform weights -> no resampling (ESS = N)."""
        initial = EmpiricalDistribution(jnp.zeros((100, 2)))
        step = with_resampling(trivial_step, ess_threshold=0.5)
        trace = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert trace[0].info["resampled"] is False
        assert trace[0].info["ess_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_resample_degenerate(self):
        """Highly non-uniform weights -> resampling triggered."""
        n = 100
        # Create degenerate weights: almost all weight on one particle
        log_w = jnp.full(n, -100.0).at[0].set(0.0)
        samples = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2)

        def weighted_step(dist, inp):
            """Return an EmpiricalDistribution with degenerate weights."""
            return EmpiricalDistribution(samples, log_weights=log_w)

        initial = EmpiricalDistribution(jnp.zeros((n, 2)))
        step = with_resampling(weighted_step, ess_threshold=0.5)
        trace = iterate(step_fn=step, initial=initial, inputs=[0.0])
        assert trace[0].info["resampled"] is True
        assert trace[0].info["ess_ratio"] < 0.5
        # Resampled distribution should have uniform weights
        resampled = trace[0].distribution
        assert resampled.is_uniform

    def test_resample_provenance(self):
        """Resampled distribution gets 'resample' provenance."""
        n = 50
        log_w = jnp.full(n, -100.0).at[0].set(0.0)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(jnp.zeros((n, 2)), log_weights=log_w)

        initial = EmpiricalDistribution(jnp.zeros((n, 2)))
        step = with_resampling(weighted_step, ess_threshold=0.5)
        trace = iterate(step_fn=step, initial=initial, inputs=[0.0])
        dist = trace[0].distribution
        assert dist.source is not None
        assert dist.source.operation == "resample"

    def test_non_empirical_passthrough(self):
        """Non-EmpiricalDistribution passes through unchanged."""
        initial = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))

        def mvn_step(dist, inp):
            return MultivariateNormal(loc=jnp.ones(2) * inp, cov=jnp.eye(2))

        step = with_resampling(mvn_step, ess_threshold=0.5)
        trace = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert isinstance(trace.final, MultivariateNormal)
        # No resampling keys in info (not an EmpiricalDistribution)
        assert "resampled" not in trace[0].info

    def test_deterministic_seed(self):
        """Resampling is deterministic across repeated calls with same seed."""
        n = 100
        log_w = jnp.full(n, -100.0).at[0].set(0.0)
        samples = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(samples, log_weights=log_w)

        initial = EmpiricalDistribution(jnp.zeros((n, 2)))

        # Run twice with the same seed — results should match
        step1 = with_resampling(weighted_step, ess_threshold=0.5, seed=42)
        trace1 = iterate(step_fn=step1, initial=initial, inputs=[0.0, 0.0])

        step2 = with_resampling(weighted_step, ess_threshold=0.5, seed=42)
        trace2 = iterate(step_fn=step2, initial=initial, inputs=[0.0, 0.0])

        assert jnp.allclose(trace1[0].distribution.samples, trace2[0].distribution.samples)
        assert jnp.allclose(trace1[1].distribution.samples, trace2[1].distribution.samples)


# ---------------------------------------------------------------------------
# Category 4: Integration — ConditioningStep + IncrementalConditioner
# ---------------------------------------------------------------------------


def _mock_condition_fn(model, data):
    """Mock conditioning: return EmpiricalDistribution near data mean."""
    data_mean = jnp.mean(jnp.asarray(data), axis=0)
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(50, data_mean.shape[0]))
    samples = data_mean[None, :] + noise * 0.1
    return EmpiricalDistribution(samples)


class TestConditioningStep:
    def test_construction(self):
        """ConditioningStep is callable and a WorkflowFunction."""
        class SimpleLikelihood:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

        step = ConditioningStep(SimpleLikelihood(), condition_fn=_mock_condition_fn)
        assert isinstance(step, WorkflowFunction)

    def test_step_returns_step_result(self):
        """Calling the step returns a StepResult."""
        class SimpleLikelihood:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

        step = ConditioningStep(SimpleLikelihood(), condition_fn=_mock_condition_fn)
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0)
        data = jnp.ones((10, 2)) * 3.0
        result = step(prior, data)
        assert isinstance(result, StepResult)
        assert isinstance(result.distribution, EmpiricalDistribution)

    def test_iterate_with_conditioning_step(self):
        """ConditioningStep works with iterate over multiple batches."""
        class SimpleLikelihood:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

        step = ConditioningStep(SimpleLikelihood(), condition_fn=_mock_condition_fn)
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0)
        batches = [jnp.ones((10, 2)) * i for i in [1.0, 2.0, 3.0]]
        trace = iterate(step, prior, batches)
        assert len(trace) == 3
        assert all(isinstance(r.distribution, EmpiricalDistribution) for r in trace.results)


class TestIncrementalConditioner:
    def test_update_single_batch(self):
        """update() conditions on a single data batch and returns StepResult."""
        class SimpleLikelihood:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0)
        conditioner = IncrementalConditioner(
            prior, SimpleLikelihood(), condition_fn=_mock_condition_fn,
        )
        data = jnp.ones((10, 2)) * 2.0
        result = conditioner.update(data=data)
        assert isinstance(result, StepResult)
        assert isinstance(result.distribution, EmpiricalDistribution)

    def test_step_property(self):
        """step property exposes the ConditioningStep for use with iterate."""
        class SimpleLikelihood:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10.0)
        conditioner = IncrementalConditioner(
            prior, SimpleLikelihood(), condition_fn=_mock_condition_fn,
        )
        assert isinstance(conditioner.step, ConditioningStep)

        # Use .step with iterate for multi-batch
        batches = [jnp.ones((10, 2)) * i for i in [1.0, 2.0]]
        trace = iterate(conditioner.step, prior, batches)
        assert len(trace) == 2
        assert all(isinstance(r.distribution, EmpiricalDistribution) for r in trace)


# ---------------------------------------------------------------------------
# Category 5: Nestability
# ---------------------------------------------------------------------------


class TestNestability:
    def test_nested_iterate(self, initial):
        """A step function can call iterate internally."""
        def inner_step(dist, value):
            samples = dist.samples + value
            return EmpiricalDistribution(samples)

        def outer_step(dist, batch):
            """Run an inner iterate loop for each outer step."""
            inner_trace = iterate(
                step_fn=inner_step, initial=dist, inputs=batch,
            )
            return StepResult(
                distribution=inner_trace.final,
                info={"inner_steps": len(inner_trace)},
            )

        outer_inputs = [[0.1, 0.2], [0.3, 0.4, 0.5]]
        trace = iterate(step_fn=outer_step, initial=initial, inputs=outer_inputs)
        assert len(trace) == 2
        assert trace[0].info["inner_steps"] == 2
        assert trace[1].info["inner_steps"] == 3
        # Total shift: (0.1+0.2) + (0.3+0.4+0.5) = 1.5
        assert jnp.allclose(trace.final.samples, jnp.full((50, 2), 1.5))
